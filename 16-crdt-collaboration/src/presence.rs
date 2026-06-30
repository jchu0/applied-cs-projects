//! Presence and awareness tracking.

use crate::{ClientId, DocumentId};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// User status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserStatus {
    /// User is actively editing.
    Active,
    /// User is idle (no activity for 30s).
    Idle,
    /// User is away (no activity for 5min).
    Away,
}

/// Cursor position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorPosition {
    /// Position in document.
    pub position: usize,
    /// Anchor for selection (same as position if no selection).
    pub anchor: usize,
}

/// Selection range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Selection {
    /// Start position.
    pub start: usize,
    /// End position.
    pub end: usize,
}

/// User presence state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceState {
    /// Client ID.
    pub client_id: ClientId,
    /// User name.
    pub name: String,
    /// User color (for cursor/selection).
    pub color: String,
    /// Cursor position.
    pub cursor: Option<CursorPosition>,
    /// Selection range.
    pub selection: Option<Selection>,
    /// Last activity timestamp.
    pub last_active: u64,
    /// User status.
    pub status: UserStatus,
}

impl PresenceState {
    /// Create new presence state.
    pub fn new(client_id: ClientId, name: String, color: String) -> Self {
        Self {
            client_id,
            name,
            color,
            cursor: None,
            selection: None,
            last_active: current_timestamp(),
            status: UserStatus::Active,
        }
    }

    /// Update cursor position.
    pub fn set_cursor(&mut self, position: usize, anchor: usize) {
        self.cursor = Some(CursorPosition { position, anchor });
        self.last_active = current_timestamp();
        self.status = UserStatus::Active;
    }

    /// Update selection.
    pub fn set_selection(&mut self, start: usize, end: usize) {
        self.selection = Some(Selection { start, end });
        self.last_active = current_timestamp();
        self.status = UserStatus::Active;
    }

    /// Update status based on inactivity.
    pub fn update_status(&mut self) {
        let now = current_timestamp();
        let inactive_duration = now.saturating_sub(self.last_active);

        self.status = if inactive_duration > 300_000 {
            // 5 minutes
            UserStatus::Away
        } else if inactive_duration > 30_000 {
            // 30 seconds
            UserStatus::Idle
        } else {
            UserStatus::Active
        };
    }
}

/// Presence tracker for a document.
pub struct DocumentPresence {
    /// Document ID.
    pub doc_id: DocumentId,
    /// User presence states.
    pub users: DashMap<ClientId, PresenceState>,
}

impl DocumentPresence {
    /// Create new document presence tracker.
    pub fn new(doc_id: DocumentId) -> Self {
        Self {
            doc_id,
            users: DashMap::new(),
        }
    }

    /// Add a user.
    pub fn add_user(&self, state: PresenceState) {
        self.users.insert(state.client_id, state);
    }

    /// Remove a user.
    pub fn remove_user(&self, client_id: &ClientId) {
        self.users.remove(client_id);
    }

    /// Update cursor position.
    pub fn update_cursor(&self, client_id: &ClientId, position: usize, anchor: usize) {
        if let Some(mut state) = self.users.get_mut(client_id) {
            state.set_cursor(position, anchor);
        }
    }

    /// Update selection.
    pub fn update_selection(&self, client_id: &ClientId, start: usize, end: usize) {
        if let Some(mut state) = self.users.get_mut(client_id) {
            state.set_selection(start, end);
        }
    }

    /// Get all presence states.
    pub fn get_all(&self) -> Vec<PresenceState> {
        self.users.iter().map(|r| r.value().clone()).collect()
    }

    /// Get a user's presence state.
    pub fn get(&self, client_id: &ClientId) -> Option<PresenceState> {
        self.users.get(client_id).map(|r| r.value().clone())
    }

    /// Update all user statuses.
    pub fn update_statuses(&self) {
        for mut state in self.users.iter_mut() {
            state.update_status();
        }
    }

    /// Get active user count.
    pub fn active_count(&self) -> usize {
        self.users
            .iter()
            .filter(|r| r.status == UserStatus::Active)
            .count()
    }

    /// Get total user count.
    pub fn total_count(&self) -> usize {
        self.users.len()
    }
}

/// Global presence manager.
pub struct PresenceManager {
    /// Presence per document.
    documents: DashMap<DocumentId, Arc<DocumentPresence>>,
}

impl PresenceManager {
    /// Create new presence manager.
    pub fn new() -> Self {
        Self {
            documents: DashMap::new(),
        }
    }

    /// Get or create document presence.
    pub fn get_or_create(&self, doc_id: DocumentId) -> Arc<DocumentPresence> {
        self.documents
            .entry(doc_id)
            .or_insert_with(|| Arc::new(DocumentPresence::new(doc_id)))
            .clone()
    }

    /// Get document presence.
    pub fn get(&self, doc_id: &DocumentId) -> Option<Arc<DocumentPresence>> {
        self.documents.get(doc_id).map(|r| r.clone())
    }

    /// Remove document presence.
    pub fn remove(&self, doc_id: &DocumentId) {
        self.documents.remove(doc_id);
    }

    /// Clean up empty documents.
    pub fn cleanup(&self) {
        self.documents.retain(|_, presence| presence.total_count() > 0);
    }
}

impl Default for PresenceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Awareness update for local client.
#[derive(Debug, Clone)]
pub struct AwarenessUpdate {
    /// Update type.
    pub update_type: AwarenessUpdateType,
    /// Affected client.
    pub client_id: ClientId,
}

/// Awareness update type.
#[derive(Debug, Clone)]
pub enum AwarenessUpdateType {
    /// User joined.
    Join(PresenceState),
    /// User left.
    Leave,
    /// User updated cursor.
    Cursor(CursorPosition),
    /// User updated selection.
    Selection(Selection),
    /// User status changed.
    Status(UserStatus),
}

/// Get current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Presence color generator.
pub struct ColorGenerator {
    colors: Vec<&'static str>,
    index: usize,
}

impl ColorGenerator {
    /// Create new color generator.
    pub fn new() -> Self {
        Self {
            colors: vec![
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
            ],
            index: 0,
        }
    }

    /// Get next color.
    pub fn next(&mut self) -> &'static str {
        let color = self.colors[self.index % self.colors.len()];
        self.index += 1;
        color
    }
}

impl Default for ColorGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presence_state() {
        let client_id = ClientId::new_v4();
        let mut state = PresenceState::new(client_id, "Test".into(), "#FF0000".into());

        state.set_cursor(10, 10);
        assert_eq!(state.cursor.unwrap().position, 10);
        assert_eq!(state.status, UserStatus::Active);
    }

    #[test]
    fn test_document_presence() {
        let doc_id = DocumentId::new_v4();
        let presence = DocumentPresence::new(doc_id);

        let client_id = ClientId::new_v4();
        let state = PresenceState::new(client_id, "User1".into(), "#00FF00".into());
        presence.add_user(state);

        assert_eq!(presence.total_count(), 1);
        assert!(presence.get(&client_id).is_some());

        presence.remove_user(&client_id);
        assert_eq!(presence.total_count(), 0);
    }
}
