// Package opa provides Open Policy Agent integration for authorization decisions.
package opa

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/open-policy-agent/opa/rego"
	"github.com/open-policy-agent/opa/storage/inmem"
)

// PolicyEngine manages OPA policies and evaluates authorization decisions.
type PolicyEngine struct {
	mu          sync.RWMutex
	queries     map[string]rego.PreparedEvalQuery
	policyDir   string
	reloadEvery time.Duration
	stopReload  chan struct{}
}

// Config holds configuration for the OPA policy engine.
type Config struct {
	PolicyDir     string        `json:"policy_dir" yaml:"policy_dir"`
	ReloadEvery   time.Duration `json:"reload_every" yaml:"reload_every"`
	EnableMetrics bool          `json:"enable_metrics" yaml:"enable_metrics"`
}

// AuthzInput represents the input for an authorization decision.
type AuthzInput struct {
	Subject    Subject    `json:"subject"`
	Action     string     `json:"action"`
	Resource   Resource   `json:"resource"`
	Context    AuthzCtx   `json:"context"`
}

// Subject represents the entity making the request.
type Subject struct {
	UserID   string   `json:"user_id"`
	TenantID string   `json:"tenant_id"`
	Roles    []string `json:"roles"`
	Groups   []string `json:"groups"`
}

// Resource represents the resource being accessed.
type Resource struct {
	Type   string            `json:"type"`
	ID     string            `json:"id"`
	Attrs  map[string]string `json:"attrs,omitempty"`
	Owner  string            `json:"owner,omitempty"`
	Tenant string            `json:"tenant,omitempty"`
}

// AuthzCtx provides additional context for authorization.
type AuthzCtx struct {
	IP        string            `json:"ip"`
	UserAgent string            `json:"user_agent"`
	Time      time.Time         `json:"time"`
	Headers   map[string]string `json:"headers,omitempty"`
}

// AuthzResult holds the result of an authorization decision.
type AuthzResult struct {
	Allow   bool     `json:"allow"`
	Reasons []string `json:"reasons,omitempty"`
	Deny    []string `json:"deny,omitempty"`
}

// NewPolicyEngine creates a new OPA policy engine.
func NewPolicyEngine(cfg Config) (*PolicyEngine, error) {
	pe := &PolicyEngine{
		queries:     make(map[string]rego.PreparedEvalQuery),
		policyDir:   cfg.PolicyDir,
		reloadEvery: cfg.ReloadEvery,
		stopReload:  make(chan struct{}),
	}

	if err := pe.loadPolicies(); err != nil {
		return nil, fmt.Errorf("failed to load policies: %w", err)
	}

	if cfg.ReloadEvery > 0 {
		go pe.reloadLoop()
	}

	return pe, nil
}

// loadPolicies loads all .rego files from the policy directory.
func (pe *PolicyEngine) loadPolicies() error {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	if pe.policyDir == "" {
		return nil
	}

	files, err := filepath.Glob(filepath.Join(pe.policyDir, "*.rego"))
	if err != nil {
		return err
	}

	var modules []func(*rego.Rego)
	for _, file := range files {
		content, err := ioutil.ReadFile(file)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", file, err)
		}
		modules = append(modules, rego.Module(filepath.Base(file), string(content)))
	}

	// Create the prepared query for authz decisions
	r := rego.New(
		rego.Query("data.authz.allow"),
		modules[0], // Base module
	)
	for i := 1; i < len(modules); i++ {
		r = rego.New(
			rego.Query("data.authz.allow"),
			modules[i],
		)
	}

	// Prepare the combined query
	combined := append([]func(*rego.Rego){
		rego.Query("data.authz"),
	}, modules...)

	authzRego := rego.New(combined...)
	query, err := authzRego.PrepareForEval(context.Background())
	if err != nil {
		return fmt.Errorf("failed to prepare authz query: %w", err)
	}

	pe.queries["authz"] = query
	return nil
}

// reloadLoop periodically reloads policies.
func (pe *PolicyEngine) reloadLoop() {
	ticker := time.NewTicker(pe.reloadEvery)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := pe.loadPolicies(); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to reload policies: %v\n", err)
			}
		case <-pe.stopReload:
			return
		}
	}
}

// Stop stops the policy engine.
func (pe *PolicyEngine) Stop() {
	close(pe.stopReload)
}

// Authorize evaluates an authorization decision.
func (pe *PolicyEngine) Authorize(ctx context.Context, input AuthzInput) (*AuthzResult, error) {
	pe.mu.RLock()
	query, ok := pe.queries["authz"]
	pe.mu.RUnlock()

	if !ok {
		// No policies loaded, default allow
		return &AuthzResult{Allow: true, Reasons: []string{"no policies configured"}}, nil
	}

	inputMap := map[string]interface{}{
		"subject":  toMap(input.Subject),
		"action":   input.Action,
		"resource": toMap(input.Resource),
		"context":  toMap(input.Context),
	}

	results, err := query.Eval(ctx, rego.EvalInput(inputMap))
	if err != nil {
		return nil, fmt.Errorf("policy evaluation failed: %w", err)
	}

	if len(results) == 0 || len(results[0].Expressions) == 0 {
		return &AuthzResult{Allow: false, Deny: []string{"no policy result"}}, nil
	}

	result := &AuthzResult{Allow: false}
	if resultMap, ok := results[0].Expressions[0].Value.(map[string]interface{}); ok {
		if allow, ok := resultMap["allow"].(bool); ok {
			result.Allow = allow
		}
		if reasons, ok := resultMap["reasons"].([]interface{}); ok {
			for _, r := range reasons {
				if s, ok := r.(string); ok {
					result.Reasons = append(result.Reasons, s)
				}
			}
		}
		if deny, ok := resultMap["deny"].([]interface{}); ok {
			for _, d := range deny {
				if s, ok := d.(string); ok {
					result.Deny = append(result.Deny, s)
				}
			}
		}
	}

	return result, nil
}

// EvalQuery evaluates a custom OPA query.
func (pe *PolicyEngine) EvalQuery(ctx context.Context, query string, input interface{}) (interface{}, error) {
	store := inmem.New()

	r := rego.New(
		rego.Query(query),
		rego.Store(store),
	)

	// Load all policies
	files, _ := filepath.Glob(filepath.Join(pe.policyDir, "*.rego"))
	for _, file := range files {
		content, _ := ioutil.ReadFile(file)
		r = rego.New(
			rego.Query(query),
			rego.Module(filepath.Base(file), string(content)),
			rego.Store(store),
		)
	}

	prepared, err := r.PrepareForEval(ctx)
	if err != nil {
		return nil, err
	}

	results, err := prepared.Eval(ctx, rego.EvalInput(input))
	if err != nil {
		return nil, err
	}

	if len(results) == 0 || len(results[0].Expressions) == 0 {
		return nil, nil
	}

	return results[0].Expressions[0].Value, nil
}

// toMap converts a struct to a map for OPA input.
func toMap(v interface{}) map[string]interface{} {
	data, _ := json.Marshal(v)
	var result map[string]interface{}
	json.Unmarshal(data, &result)
	return result
}

// DefaultPolicies returns embedded default policies for common scenarios.
func DefaultPolicies() map[string]string {
	return map[string]string{
		"authz.rego":   defaultAuthzPolicy,
		"rbac.rego":    defaultRBACPolicy,
		"tenant.rego":  defaultTenantPolicy,
		"rate.rego":    defaultRateLimitPolicy,
	}
}

const defaultAuthzPolicy = `package authz

import future.keywords.if
import future.keywords.in

default allow := false

# Allow if RBAC permits
allow if {
    rbac_allow
}

# Allow if user is tenant admin
allow if {
    is_tenant_admin
}

rbac_allow if {
    some role in input.subject.roles
    some perm in data.rbac.role_permissions[role]
    perm.action == input.action
    perm.resource == input.resource.type
}

is_tenant_admin if {
    "tenant_admin" in input.subject.roles
    input.subject.tenant_id == input.resource.tenant
}

# Collect reasons for allow
reasons[msg] if {
    rbac_allow
    msg := "RBAC permission granted"
}

reasons[msg] if {
    is_tenant_admin
    msg := "Tenant admin access"
}

# Collect deny reasons
deny[msg] if {
    not allow
    msg := "No matching permission found"
}
`

const defaultRBACPolicy = `package rbac

# Role definitions
role_permissions := {
    "admin": [
        {"action": "read", "resource": "user"},
        {"action": "write", "resource": "user"},
        {"action": "delete", "resource": "user"},
        {"action": "read", "resource": "billing"},
        {"action": "write", "resource": "billing"},
        {"action": "read", "resource": "settings"},
        {"action": "write", "resource": "settings"},
    ],
    "user": [
        {"action": "read", "resource": "user"},
        {"action": "read", "resource": "billing"},
    ],
    "billing_admin": [
        {"action": "read", "resource": "billing"},
        {"action": "write", "resource": "billing"},
        {"action": "delete", "resource": "billing"},
    ],
    "readonly": [
        {"action": "read", "resource": "user"},
        {"action": "read", "resource": "billing"},
        {"action": "read", "resource": "settings"},
    ],
}
`

const defaultTenantPolicy = `package tenant

import future.keywords.if
import future.keywords.in

# Enforce tenant isolation
default tenant_allowed := false

tenant_allowed if {
    input.subject.tenant_id == input.resource.tenant
}

tenant_allowed if {
    # System users can access any tenant
    "system" in input.subject.roles
}

# Cross-tenant access is denied by default
deny[msg] if {
    not tenant_allowed
    msg := sprintf("Cross-tenant access denied: user tenant %s != resource tenant %s",
                   [input.subject.tenant_id, input.resource.tenant])
}
`

const defaultRateLimitPolicy = `package rate_limit

import future.keywords.if
import future.keywords.in

# Default rate limits per role
limits := {
    "admin": {"requests_per_minute": 1000, "requests_per_hour": 10000},
    "user": {"requests_per_minute": 100, "requests_per_hour": 1000},
    "guest": {"requests_per_minute": 10, "requests_per_hour": 100},
}

# Get rate limit for user
rate_limit := limit if {
    some role in input.subject.roles
    limit := limits[role]
}

# Default limit for unknown roles
rate_limit := {"requests_per_minute": 10, "requests_per_hour": 100} if {
    not any_known_role
}

any_known_role if {
    some role in input.subject.roles
    limits[role]
}
`

// WriteDefaultPolicies writes the default policies to a directory.
func WriteDefaultPolicies(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	for name, content := range DefaultPolicies() {
		path := filepath.Join(dir, name)
		if err := ioutil.WriteFile(path, []byte(content), 0644); err != nil {
			return err
		}
	}

	return nil
}
