package com.example.desktopapp;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class ConvoManager {
    private static final String DB_URL = "jdbc:sqlite:chat_history.db";
    private int currentConversationId = -1;

    public static class Message {
        public final String author;
        public final String content;
        public final long   timestamp;
        public Message(String author, String content, long timestamp) {
            this.author    = author;
            this.content   = content;
            this.timestamp = timestamp;
        }
    }

    public static class Conversation {
        public final int    id;
        public final String title;
        public final long   created;
        public Conversation(int id, String title, long created) {
            this.id      = id;
            this.title   = title;
            this.created = created;
        }
        @Override public String toString() { return title; }
    }

    public void init() {
        try (Connection c = connect(); Statement s = c.createStatement()) {
            s.execute(
                "CREATE TABLE IF NOT EXISTS conversations (" +
                "    id      INTEGER PRIMARY KEY AUTOINCREMENT," +
                "    title   TEXT    NOT NULL DEFAULT 'New Chat'," +
                "    created INTEGER NOT NULL" +
                ")"
            );
            s.execute(
                "CREATE TABLE IF NOT EXISTS messages (" +
                "    id              INTEGER PRIMARY KEY AUTOINCREMENT," +
                "    conversation_id INTEGER NOT NULL," +
                "    author          TEXT    NOT NULL," +
                "    content         TEXT    NOT NULL," +
                "    timestamp       INTEGER NOT NULL" +
                ")"
            );
        } catch (SQLException e) {
            System.err.println("[ConvoManager] init failed: " + e.getMessage());
        }
        // Load most recent conversation or create one
        List<Conversation> all = loadAllConversations();
        if (all.isEmpty()) {
            newConversation();
        } else {
            currentConversationId = all.get(0).id;
        }
    }

    public int newConversation() {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                "INSERT INTO conversations (title, created) VALUES (?, ?)",
                Statement.RETURN_GENERATED_KEYS)) {
            ps.setString(1, "New Chat");
            ps.setLong(2, System.currentTimeMillis());
            ps.executeUpdate();
            ResultSet keys = ps.getGeneratedKeys();
            if (keys.next()) currentConversationId = keys.getInt(1);
        } catch (SQLException e) {
            System.err.println("[ConvoManager] newConversation failed: " + e.getMessage());
        }
        return currentConversationId;
    }

    public void setCurrentConversation(int id) {
        currentConversationId = id;
    }

    public int getCurrentConversationId() {
        return currentConversationId;
    }

    /** Updates the title of the current conversation using the first user message. */
    public void updateTitle(String firstUserMessage) {
        String title = firstUserMessage.length() > 40
            ? firstUserMessage.substring(0, 40) + "..."
            : firstUserMessage;
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                "UPDATE conversations SET title = ? WHERE id = ?")) {
            ps.setString(1, title);
            ps.setInt(2, currentConversationId);
            ps.executeUpdate();
        } catch (SQLException e) {
            System.err.println("[ConvoManager] updateTitle failed: " + e.getMessage());
        }
    }

    public void saveMessage(String author, String content) {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                "INSERT INTO messages (conversation_id, author, content, timestamp) VALUES (?, ?, ?, ?)")) {
            ps.setInt(1, currentConversationId);
            ps.setString(2, author);
            ps.setString(3, content);
            ps.setLong(4, System.currentTimeMillis());
            ps.executeUpdate();
        } catch (SQLException e) {
            System.err.println("[ConvoManager] save failed: " + e.getMessage());
        }
    }

    public List<Message> loadHistory() {
        List<Message> history = new ArrayList();
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                "SELECT author, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC")) {
            ps.setInt(1, currentConversationId);
            ResultSet rs = ps.executeQuery();
            while (rs.next()) {
                history.add(new Message(
                    rs.getString("author"),
                    rs.getString("content"),
                    rs.getLong("timestamp")
                ));
            }
        } catch (SQLException e) {
            System.err.println("[ConvoManager] load failed: " + e.getMessage());
        }
        return history;
    }

    public List<Conversation> loadAllConversations() {
        List<Conversation> list = new ArrayList();
        try (Connection c = connect();
             Statement s = c.createStatement();
             ResultSet rs = s.executeQuery(
                "SELECT id, title, created FROM conversations ORDER BY created DESC")) {
            while (rs.next()) {
                list.add(new Conversation(
                    rs.getInt("id"),
                    rs.getString("title"),
                    rs.getLong("created")
                ));
            }
        } catch (SQLException e) {
            System.err.println("[ConvoManager] loadAll failed: " + e.getMessage());
        }
        return list;
    }

    public void clearHistory() {
        try (Connection c = connect();
             PreparedStatement ps = c.prepareStatement(
                "DELETE FROM messages WHERE conversation_id = ?")) {
            ps.setInt(1, currentConversationId);
            ps.executeUpdate();
        } catch (SQLException e) {
            System.err.println("[ConvoManager] clear failed: " + e.getMessage());
        }
    }

    public void deleteConversation(int id) {
        try (Connection c = connect()) {
            PreparedStatement ps1 = c.prepareStatement("DELETE FROM messages WHERE conversation_id = ?");
            ps1.setInt(1, id);
            ps1.executeUpdate();
            PreparedStatement ps2 = c.prepareStatement("DELETE FROM conversations WHERE id = ?");
            ps2.setInt(1, id);
            ps2.executeUpdate();
        } catch (SQLException e) {
            System.err.println("[ConvoManager] delete failed: " + e.getMessage());
        }
    }

    private Connection connect() throws SQLException {
        return DriverManager.getConnection(DB_URL);
    }
}
