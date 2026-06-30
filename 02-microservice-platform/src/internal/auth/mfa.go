package auth

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base32"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/mlai/microservice-platform/pkg/logging"
	"go.uber.org/zap"
)

// MFA errors
var (
	ErrMFANotEnabled      = errors.New("mfa not enabled for user")
	ErrMFAAlreadyEnabled  = errors.New("mfa already enabled for user")
	ErrMFAInvalidCode     = errors.New("invalid mfa code")
	ErrMFASetupRequired   = errors.New("mfa setup required")
	ErrMFARecoveryUsed    = errors.New("recovery code already used")
	ErrMFANoRecoveryCodes = errors.New("no recovery codes available")
)

// MFAMethod represents the MFA method type
type MFAMethod string

const (
	MFAMethodTOTP     MFAMethod = "totp"
	MFAMethodSMS      MFAMethod = "sms"
	MFAMethodEmail    MFAMethod = "email"
	MFAMethodRecovery MFAMethod = "recovery"
)

// MFAConfig holds MFA configuration
type MFAConfig struct {
	Issuer           string
	TOTPDigits       int
	TOTPPeriod       int
	TOTPSkew         int // Number of periods to allow for clock skew
	RecoveryCodeLen  int
	RecoveryCodeNum  int
	SMSProvider      string
	EmailProvider    string
}

// DefaultMFAConfig returns default MFA configuration
func DefaultMFAConfig() *MFAConfig {
	return &MFAConfig{
		Issuer:          "MicroservicePlatform",
		TOTPDigits:      6,
		TOTPPeriod:      30,
		TOTPSkew:        1,
		RecoveryCodeLen: 8,
		RecoveryCodeNum: 10,
	}
}

// MFASetup contains MFA setup information
type MFASetup struct {
	Secret        string   `json:"secret"`
	QRCodeURL     string   `json:"qr_code_url"`
	RecoveryCodes []string `json:"recovery_codes"`
}

// MFAStatus represents the MFA status for a user
type MFAStatus struct {
	Enabled        bool      `json:"enabled"`
	Method         MFAMethod `json:"method"`
	SetupAt        time.Time `json:"setup_at,omitempty"`
	LastUsedAt     time.Time `json:"last_used_at,omitempty"`
	RecoveryCodesLeft int    `json:"recovery_codes_left"`
}

// MFAChallenge represents an MFA challenge
type MFAChallenge struct {
	ChallengeID string    `json:"challenge_id"`
	Method      MFAMethod `json:"method"`
	ExpiresAt   time.Time `json:"expires_at"`
	Hint        string    `json:"hint,omitempty"` // e.g., last 4 digits of phone
}

// MFAManager handles multi-factor authentication
type MFAManager struct {
	config  *MFAConfig
	store   MFAStore
	logger  *logging.Logger
}

// MFAStore interface for storing MFA data
type MFAStore interface {
	// GetMFASecret retrieves the MFA secret for a user
	GetMFASecret(ctx context.Context, userID, tenantID string) (string, error)
	// SetMFASecret stores the MFA secret for a user
	SetMFASecret(ctx context.Context, userID, tenantID, secret string) error
	// DeleteMFASecret removes the MFA secret for a user
	DeleteMFASecret(ctx context.Context, userID, tenantID string) error
	// GetRecoveryCodes retrieves recovery codes for a user
	GetRecoveryCodes(ctx context.Context, userID, tenantID string) ([]string, error)
	// SetRecoveryCodes stores recovery codes for a user
	SetRecoveryCodes(ctx context.Context, userID, tenantID string, codes []string) error
	// MarkRecoveryCodeUsed marks a recovery code as used
	MarkRecoveryCodeUsed(ctx context.Context, userID, tenantID, code string) error
	// IsMFAEnabled checks if MFA is enabled for a user
	IsMFAEnabled(ctx context.Context, userID, tenantID string) (bool, error)
	// SetMFAEnabled enables/disables MFA for a user
	SetMFAEnabled(ctx context.Context, userID, tenantID string, enabled bool) error
}

// NewMFAManager creates a new MFA manager
func NewMFAManager(config *MFAConfig, store MFAStore, logger *logging.Logger) *MFAManager {
	if config == nil {
		config = DefaultMFAConfig()
	}
	return &MFAManager{
		config: config,
		store:  store,
		logger: logger,
	}
}

// InitiateSetup starts the MFA setup process for a user
func (m *MFAManager) InitiateSetup(ctx context.Context, userID, tenantID, email string) (*MFASetup, error) {
	logger := m.logger.WithContext(ctx)

	// Check if MFA is already enabled
	enabled, err := m.store.IsMFAEnabled(ctx, userID, tenantID)
	if err != nil {
		return nil, fmt.Errorf("failed to check mfa status: %w", err)
	}
	if enabled {
		return nil, ErrMFAAlreadyEnabled
	}

	// Generate secret
	secret, err := generateTOTPSecret()
	if err != nil {
		return nil, fmt.Errorf("failed to generate secret: %w", err)
	}

	// Store secret (not yet confirmed)
	if err := m.store.SetMFASecret(ctx, userID, tenantID, secret); err != nil {
		return nil, fmt.Errorf("failed to store secret: %w", err)
	}

	// Generate recovery codes
	recoveryCodes := m.generateRecoveryCodes()
	if err := m.store.SetRecoveryCodes(ctx, userID, tenantID, recoveryCodes); err != nil {
		return nil, fmt.Errorf("failed to store recovery codes: %w", err)
	}

	// Generate QR code URL
	qrCodeURL := m.generateQRCodeURL(secret, email)

	logger.Info("mfa setup initiated",
		zap.String("user_id", userID),
	)

	return &MFASetup{
		Secret:        secret,
		QRCodeURL:     qrCodeURL,
		RecoveryCodes: recoveryCodes,
	}, nil
}

// ConfirmSetup confirms and enables MFA for a user
func (m *MFAManager) ConfirmSetup(ctx context.Context, userID, tenantID, code string) error {
	logger := m.logger.WithContext(ctx)

	// Validate the code to ensure user has correctly set up their authenticator
	valid, err := m.ValidateTOTP(ctx, userID, tenantID, code)
	if err != nil {
		return err
	}
	if !valid {
		return ErrMFAInvalidCode
	}

	// Enable MFA
	if err := m.store.SetMFAEnabled(ctx, userID, tenantID, true); err != nil {
		return fmt.Errorf("failed to enable mfa: %w", err)
	}

	logger.Info("mfa enabled",
		zap.String("user_id", userID),
	)

	return nil
}

// Disable disables MFA for a user
func (m *MFAManager) Disable(ctx context.Context, userID, tenantID, code string) error {
	logger := m.logger.WithContext(ctx)

	// Require valid code to disable
	valid, err := m.ValidateTOTP(ctx, userID, tenantID, code)
	if err != nil {
		return err
	}
	if !valid {
		return ErrMFAInvalidCode
	}

	// Disable MFA
	if err := m.store.SetMFAEnabled(ctx, userID, tenantID, false); err != nil {
		return fmt.Errorf("failed to disable mfa: %w", err)
	}

	// Delete secret and recovery codes
	if err := m.store.DeleteMFASecret(ctx, userID, tenantID); err != nil {
		logger.Warn("failed to delete mfa secret", zap.Error(err))
	}

	logger.Info("mfa disabled",
		zap.String("user_id", userID),
	)

	return nil
}

// ValidateTOTP validates a TOTP code
func (m *MFAManager) ValidateTOTP(ctx context.Context, userID, tenantID, code string) (bool, error) {
	// Get secret
	secret, err := m.store.GetMFASecret(ctx, userID, tenantID)
	if err != nil {
		return false, ErrMFANotEnabled
	}

	// Validate code with clock skew tolerance
	now := time.Now().Unix()
	period := int64(m.config.TOTPPeriod)

	for i := -m.config.TOTPSkew; i <= m.config.TOTPSkew; i++ {
		counter := (now / period) + int64(i)
		expectedCode := generateTOTPCode(secret, counter, m.config.TOTPDigits)
		if hmac.Equal([]byte(code), []byte(expectedCode)) {
			return true, nil
		}
	}

	return false, nil
}

// ValidateRecoveryCode validates and consumes a recovery code
func (m *MFAManager) ValidateRecoveryCode(ctx context.Context, userID, tenantID, code string) (bool, error) {
	logger := m.logger.WithContext(ctx)

	// Get recovery codes
	codes, err := m.store.GetRecoveryCodes(ctx, userID, tenantID)
	if err != nil {
		return false, err
	}

	// Normalize code (remove dashes, uppercase)
	normalizedCode := strings.ToUpper(strings.ReplaceAll(code, "-", ""))

	// Check if code is valid and unused
	for _, c := range codes {
		normalizedStored := strings.ToUpper(strings.ReplaceAll(c, "-", ""))
		if normalizedStored == normalizedCode {
			// Mark as used
			if err := m.store.MarkRecoveryCodeUsed(ctx, userID, tenantID, c); err != nil {
				return false, err
			}
			logger.Info("recovery code used",
				zap.String("user_id", userID),
			)
			return true, nil
		}
	}

	return false, nil
}

// GetStatus returns the MFA status for a user
func (m *MFAManager) GetStatus(ctx context.Context, userID, tenantID string) (*MFAStatus, error) {
	enabled, err := m.store.IsMFAEnabled(ctx, userID, tenantID)
	if err != nil {
		return nil, err
	}

	codes, _ := m.store.GetRecoveryCodes(ctx, userID, tenantID)

	return &MFAStatus{
		Enabled:           enabled,
		Method:            MFAMethodTOTP,
		RecoveryCodesLeft: len(codes),
	}, nil
}

// RegenerateRecoveryCodes generates new recovery codes
func (m *MFAManager) RegenerateRecoveryCodes(ctx context.Context, userID, tenantID, code string) ([]string, error) {
	logger := m.logger.WithContext(ctx)

	// Require valid TOTP code
	valid, err := m.ValidateTOTP(ctx, userID, tenantID, code)
	if err != nil {
		return nil, err
	}
	if !valid {
		return nil, ErrMFAInvalidCode
	}

	// Generate new codes
	newCodes := m.generateRecoveryCodes()
	if err := m.store.SetRecoveryCodes(ctx, userID, tenantID, newCodes); err != nil {
		return nil, fmt.Errorf("failed to store recovery codes: %w", err)
	}

	logger.Info("recovery codes regenerated",
		zap.String("user_id", userID),
	)

	return newCodes, nil
}

// generateTOTPSecret generates a random TOTP secret
func generateTOTPSecret() (string, error) {
	secret := make([]byte, 20)
	if _, err := rand.Read(secret); err != nil {
		return "", err
	}
	return base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(secret), nil
}

// generateTOTPCode generates a TOTP code for a given counter
func generateTOTPCode(secret string, counter int64, digits int) string {
	// Decode secret
	key, err := base32.StdEncoding.WithPadding(base32.NoPadding).DecodeString(secret)
	if err != nil {
		return ""
	}

	// Convert counter to bytes (big-endian)
	counterBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(counterBytes, uint64(counter))

	// Generate HMAC-SHA1
	h := hmac.New(sha1.New, key)
	h.Write(counterBytes)
	hash := h.Sum(nil)

	// Dynamic truncation
	offset := hash[len(hash)-1] & 0x0F
	truncated := binary.BigEndian.Uint32(hash[offset:offset+4]) & 0x7FFFFFFF

	// Generate code
	code := truncated % pow10(digits)
	return fmt.Sprintf("%0*d", digits, code)
}

// pow10 returns 10^n
func pow10(n int) uint32 {
	result := uint32(1)
	for i := 0; i < n; i++ {
		result *= 10
	}
	return result
}

// generateQRCodeURL generates a QR code URL for TOTP setup
func (m *MFAManager) generateQRCodeURL(secret, email string) string {
	// Format: otpauth://totp/{issuer}:{email}?secret={secret}&issuer={issuer}&digits={digits}&period={period}
	return fmt.Sprintf(
		"otpauth://totp/%s:%s?secret=%s&issuer=%s&digits=%d&period=%d",
		m.config.Issuer,
		email,
		secret,
		m.config.Issuer,
		m.config.TOTPDigits,
		m.config.TOTPPeriod,
	)
}

// generateRecoveryCodes generates recovery codes
func (m *MFAManager) generateRecoveryCodes() []string {
	codes := make([]string, m.config.RecoveryCodeNum)
	for i := 0; i < m.config.RecoveryCodeNum; i++ {
		codes[i] = generateRecoveryCode(m.config.RecoveryCodeLen)
	}
	return codes
}

// generateRecoveryCode generates a single recovery code
func generateRecoveryCode(length int) string {
	const charset = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" // Exclude confusing chars
	code := make([]byte, length)
	for i := range code {
		b := make([]byte, 1)
		rand.Read(b)
		code[i] = charset[int(b[0])%len(charset)]
	}
	// Format with dash in middle
	if length >= 4 {
		mid := length / 2
		return string(code[:mid]) + "-" + string(code[mid:])
	}
	return string(code)
}

// generateSecureToken generates a secure random token
func generateSecureToken(length int) string {
	b := make([]byte, length)
	rand.Read(b)
	h := sha256.Sum256(b)
	return base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(h[:])[:length]
}

// InMemoryMFAStore is an in-memory implementation of MFAStore for testing
type InMemoryMFAStore struct {
	secrets       map[string]string
	recoveryCodes map[string][]string
	enabled       map[string]bool
}

// NewInMemoryMFAStore creates a new in-memory MFA store
func NewInMemoryMFAStore() *InMemoryMFAStore {
	return &InMemoryMFAStore{
		secrets:       make(map[string]string),
		recoveryCodes: make(map[string][]string),
		enabled:       make(map[string]bool),
	}
}

func (s *InMemoryMFAStore) key(userID, tenantID string) string {
	return tenantID + ":" + userID
}

func (s *InMemoryMFAStore) GetMFASecret(ctx context.Context, userID, tenantID string) (string, error) {
	secret, ok := s.secrets[s.key(userID, tenantID)]
	if !ok {
		return "", ErrMFANotEnabled
	}
	return secret, nil
}

func (s *InMemoryMFAStore) SetMFASecret(ctx context.Context, userID, tenantID, secret string) error {
	s.secrets[s.key(userID, tenantID)] = secret
	return nil
}

func (s *InMemoryMFAStore) DeleteMFASecret(ctx context.Context, userID, tenantID string) error {
	delete(s.secrets, s.key(userID, tenantID))
	return nil
}

func (s *InMemoryMFAStore) GetRecoveryCodes(ctx context.Context, userID, tenantID string) ([]string, error) {
	codes, ok := s.recoveryCodes[s.key(userID, tenantID)]
	if !ok {
		return nil, nil
	}
	return codes, nil
}

func (s *InMemoryMFAStore) SetRecoveryCodes(ctx context.Context, userID, tenantID string, codes []string) error {
	s.recoveryCodes[s.key(userID, tenantID)] = codes
	return nil
}

func (s *InMemoryMFAStore) MarkRecoveryCodeUsed(ctx context.Context, userID, tenantID, code string) error {
	codes := s.recoveryCodes[s.key(userID, tenantID)]
	newCodes := make([]string, 0, len(codes))
	for _, c := range codes {
		if c != code {
			newCodes = append(newCodes, c)
		}
	}
	s.recoveryCodes[s.key(userID, tenantID)] = newCodes
	return nil
}

func (s *InMemoryMFAStore) IsMFAEnabled(ctx context.Context, userID, tenantID string) (bool, error) {
	return s.enabled[s.key(userID, tenantID)], nil
}

func (s *InMemoryMFAStore) SetMFAEnabled(ctx context.Context, userID, tenantID string, enabled bool) error {
	s.enabled[s.key(userID, tenantID)] = enabled
	return nil
}
