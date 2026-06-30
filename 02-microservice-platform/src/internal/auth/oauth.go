package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/mlai/microservice-platform/pkg/logging"
	"go.uber.org/zap"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/github"
	"golang.org/x/oauth2/google"
)

// OAuth errors
var (
	ErrOAuthProviderNotConfigured = errors.New("oauth provider not configured")
	ErrOAuthStateMismatch         = errors.New("oauth state mismatch")
	ErrOAuthEmailNotVerified      = errors.New("email not verified by oauth provider")
	ErrOAuthTokenExchange         = errors.New("failed to exchange oauth token")
)

// OAuthProvider represents an OAuth provider
type OAuthProvider string

const (
	OAuthProviderGoogle OAuthProvider = "google"
	OAuthProviderGitHub OAuthProvider = "github"
)

// OAuthConfig holds OAuth configuration for all providers
type OAuthConfig struct {
	Google *ProviderConfig
	GitHub *ProviderConfig
}

// ProviderConfig holds configuration for a single OAuth provider
type ProviderConfig struct {
	ClientID     string
	ClientSecret string
	RedirectURL  string
	Scopes       []string
}

// OAuthUserInfo contains user info from OAuth provider
type OAuthUserInfo struct {
	ID            string
	Email         string
	EmailVerified bool
	Name          string
	FirstName     string
	LastName      string
	AvatarURL     string
	Provider      OAuthProvider
	ProviderID    string
}

// OAuthManager handles OAuth authentication flows
type OAuthManager struct {
	configs map[OAuthProvider]*oauth2.Config
	logger  *logging.Logger
}

// NewOAuthManager creates a new OAuth manager
func NewOAuthManager(config *OAuthConfig, logger *logging.Logger) *OAuthManager {
	manager := &OAuthManager{
		configs: make(map[OAuthProvider]*oauth2.Config),
		logger:  logger,
	}

	// Configure Google OAuth
	if config.Google != nil && config.Google.ClientID != "" {
		scopes := config.Google.Scopes
		if len(scopes) == 0 {
			scopes = []string{
				"https://www.googleapis.com/auth/userinfo.email",
				"https://www.googleapis.com/auth/userinfo.profile",
			}
		}
		manager.configs[OAuthProviderGoogle] = &oauth2.Config{
			ClientID:     config.Google.ClientID,
			ClientSecret: config.Google.ClientSecret,
			RedirectURL:  config.Google.RedirectURL,
			Scopes:       scopes,
			Endpoint:     google.Endpoint,
		}
	}

	// Configure GitHub OAuth
	if config.GitHub != nil && config.GitHub.ClientID != "" {
		scopes := config.GitHub.Scopes
		if len(scopes) == 0 {
			scopes = []string{"user:email", "read:user"}
		}
		manager.configs[OAuthProviderGitHub] = &oauth2.Config{
			ClientID:     config.GitHub.ClientID,
			ClientSecret: config.GitHub.ClientSecret,
			RedirectURL:  config.GitHub.RedirectURL,
			Scopes:       scopes,
			Endpoint:     github.Endpoint,
		}
	}

	return manager
}

// IsProviderConfigured checks if a provider is configured
func (m *OAuthManager) IsProviderConfigured(provider OAuthProvider) bool {
	_, ok := m.configs[provider]
	return ok
}

// GetAuthURL returns the OAuth authorization URL for a provider
func (m *OAuthManager) GetAuthURL(provider OAuthProvider, state string) (string, error) {
	config, ok := m.configs[provider]
	if !ok {
		return "", ErrOAuthProviderNotConfigured
	}

	return config.AuthCodeURL(state, oauth2.AccessTypeOffline), nil
}

// ExchangeCode exchanges an authorization code for tokens and user info
func (m *OAuthManager) ExchangeCode(ctx context.Context, provider OAuthProvider, code string) (*OAuthUserInfo, error) {
	config, ok := m.configs[provider]
	if !ok {
		return nil, ErrOAuthProviderNotConfigured
	}

	// Exchange code for token
	token, err := config.Exchange(ctx, code)
	if err != nil {
		m.logger.Error("failed to exchange oauth code",
			zap.String("provider", string(provider)),
			zap.Error(err),
		)
		return nil, fmt.Errorf("%w: %v", ErrOAuthTokenExchange, err)
	}

	// Get user info based on provider
	switch provider {
	case OAuthProviderGoogle:
		return m.getGoogleUserInfo(ctx, token)
	case OAuthProviderGitHub:
		return m.getGitHubUserInfo(ctx, token)
	default:
		return nil, ErrOAuthProviderNotConfigured
	}
}

// getGoogleUserInfo fetches user info from Google
func (m *OAuthManager) getGoogleUserInfo(ctx context.Context, token *oauth2.Token) (*OAuthUserInfo, error) {
	client := oauth2.NewClient(ctx, oauth2.StaticTokenSource(token))

	resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo")
	if err != nil {
		return nil, fmt.Errorf("failed to get google user info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("google user info request failed with status: %d", resp.StatusCode)
	}

	var info struct {
		ID            string `json:"id"`
		Email         string `json:"email"`
		VerifiedEmail bool   `json:"verified_email"`
		Name          string `json:"name"`
		GivenName     string `json:"given_name"`
		FamilyName    string `json:"family_name"`
		Picture       string `json:"picture"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("failed to decode google user info: %w", err)
	}

	return &OAuthUserInfo{
		ID:            info.ID,
		Email:         info.Email,
		EmailVerified: info.VerifiedEmail,
		Name:          info.Name,
		FirstName:     info.GivenName,
		LastName:      info.FamilyName,
		AvatarURL:     info.Picture,
		Provider:      OAuthProviderGoogle,
		ProviderID:    info.ID,
	}, nil
}

// getGitHubUserInfo fetches user info from GitHub
func (m *OAuthManager) getGitHubUserInfo(ctx context.Context, token *oauth2.Token) (*OAuthUserInfo, error) {
	client := oauth2.NewClient(ctx, oauth2.StaticTokenSource(token))

	// Get user profile
	resp, err := client.Get("https://api.github.com/user")
	if err != nil {
		return nil, fmt.Errorf("failed to get github user info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("github user info request failed with status: %d", resp.StatusCode)
	}

	var info struct {
		ID        int64  `json:"id"`
		Login     string `json:"login"`
		Name      string `json:"name"`
		Email     string `json:"email"`
		AvatarURL string `json:"avatar_url"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("failed to decode github user info: %w", err)
	}

	// If email is not in profile, fetch from emails endpoint
	email := info.Email
	if email == "" {
		email, err = m.getGitHubPrimaryEmail(ctx, client)
		if err != nil {
			m.logger.Warn("failed to get github primary email", zap.Error(err))
		}
	}

	// Parse name into first/last
	firstName, lastName := parseName(info.Name)

	return &OAuthUserInfo{
		ID:            fmt.Sprintf("%d", info.ID),
		Email:         email,
		EmailVerified: email != "", // GitHub emails are verified
		Name:          info.Name,
		FirstName:     firstName,
		LastName:      lastName,
		AvatarURL:     info.AvatarURL,
		Provider:      OAuthProviderGitHub,
		ProviderID:    fmt.Sprintf("%d", info.ID),
	}, nil
}

// getGitHubPrimaryEmail fetches the primary verified email from GitHub
func (m *OAuthManager) getGitHubPrimaryEmail(ctx context.Context, client *http.Client) (string, error) {
	resp, err := client.Get("https://api.github.com/user/emails")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("github emails request failed with status: %d", resp.StatusCode)
	}

	var emails []struct {
		Email    string `json:"email"`
		Primary  bool   `json:"primary"`
		Verified bool   `json:"verified"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&emails); err != nil {
		return "", err
	}

	// Find primary verified email
	for _, e := range emails {
		if e.Primary && e.Verified {
			return e.Email, nil
		}
	}

	// Fallback to any verified email
	for _, e := range emails {
		if e.Verified {
			return e.Email, nil
		}
	}

	return "", errors.New("no verified email found")
}

// parseName splits a full name into first and last name
func parseName(name string) (string, string) {
	if name == "" {
		return "", ""
	}

	// Simple split on first space
	for i, c := range name {
		if c == ' ' {
			return name[:i], name[i+1:]
		}
	}
	return name, ""
}

// OAuthState represents the state passed during OAuth flow
type OAuthState struct {
	Nonce     string    `json:"nonce"`
	TenantID  string    `json:"tenant_id"`
	ReturnURL string    `json:"return_url"`
	CreatedAt time.Time `json:"created_at"`
}

// OAuthStateStore manages OAuth state tokens
type OAuthStateStore struct {
	states map[string]*OAuthState
	ttl    time.Duration
}

// NewOAuthStateStore creates a new OAuth state store
func NewOAuthStateStore(ttl time.Duration) *OAuthStateStore {
	return &OAuthStateStore{
		states: make(map[string]*OAuthState),
		ttl:    ttl,
	}
}

// Create creates a new OAuth state
func (s *OAuthStateStore) Create(tenantID, returnURL string) string {
	nonce := generateSecureToken(32)
	state := &OAuthState{
		Nonce:     nonce,
		TenantID:  tenantID,
		ReturnURL: returnURL,
		CreatedAt: time.Now(),
	}
	s.states[nonce] = state
	return nonce
}

// Validate validates and consumes an OAuth state
func (s *OAuthStateStore) Validate(nonce string) (*OAuthState, error) {
	state, ok := s.states[nonce]
	if !ok {
		return nil, ErrOAuthStateMismatch
	}

	// Check TTL
	if time.Since(state.CreatedAt) > s.ttl {
		delete(s.states, nonce)
		return nil, ErrOAuthStateMismatch
	}

	// Consume state (one-time use)
	delete(s.states, nonce)

	return state, nil
}

// Cleanup removes expired states
func (s *OAuthStateStore) Cleanup() {
	now := time.Now()
	for nonce, state := range s.states {
		if now.Sub(state.CreatedAt) > s.ttl {
			delete(s.states, nonce)
		}
	}
}
