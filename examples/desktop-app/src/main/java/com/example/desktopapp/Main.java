package com.example.desktopapp;

import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatLightLaf;
import com.llama4aj;

import javax.swing.*;
import javax.swing.text.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main {

    private static final String MODEL_PATH = "./model.gguf"; // !!! CHANGE THIS !!!

    private static final Color CODE_BG_LIGHT = new Color(235, 235, 235);
    private static final Color CODE_BG_DARK  = new Color(45,  45,  45);

    private boolean          darkTheme    = false;
    private volatile boolean isGenerating = false;
    private boolean          firstMessage = true;

    private llama4aj        model;
    private ExecutorService executor;
    private StringBuilder   currentResponse = new StringBuilder();
    private volatile CountDownLatch generationLatch;
    private int             streamStart = -1;

    private final ConvoManager convo = new ConvoManager();

    private JFrame    frame;
    private JTextPane chatPane;
    private JTextArea inputArea;
    private JButton   sendButton;
    private JButton   stopButton;
    private JButton   themeButton;
    private JButton   clearButton;
    private JButton   newConvoButton;
    private JLabel    statusLabel;
    private DefaultListModel<ConvoManager.Conversation> sidebarModel;
    private JList<ConvoManager.Conversation>            sidebarList;

    public static void main(String[] args) {
        FlatLightLaf.setup();
        SwingUtilities.invokeLater(() -> new Main().start());
    }

    private void start() {
        convo.init();
        buildUI();
        refreshSidebar();
        loadHistory();
        executor = Executors.newSingleThreadExecutor();
        executor.submit(this::initModel);
    }

    private void initModel() {
        setStatus("Loading model...");
        File f = new File(MODEL_PATH);
        if (!f.exists()) {
            setStatus("ERROR: model not found at " + MODEL_PATH);
            setSendEnabled(false);
            return;
        }
        try {
            model = llama4aj.load(MODEL_PATH);
            if (model != null) {
                setStatus("Ready");
                setSendEnabled(true);
            } else {
                setStatus("ERROR: failed to load model");
                setSendEnabled(false);
            }
        } catch (Exception e) {
            setStatus("ERROR: " + e.getMessage());
            setSendEnabled(false);
        }
    }

    private void sendMessage() {
        String prompt = inputArea.getText().trim();
        if (prompt.isEmpty() || isGenerating) return;
        inputArea.setText("");
        if (firstMessage) {
            convo.updateTitle(prompt);
            firstMessage = false;
            refreshSidebar();
        }
        appendMessage("You", prompt);
        convo.saveMessage("You", prompt);
        setGeneratingState(true);
        executor.submit(() -> generateResponse(prompt));
    }

    private void generateResponse(String userPrompt) {
        if (model == null) {
            setStatus("ERROR: model not loaded");
            setGeneratingState(false);
            return;
        }

        currentResponse = new StringBuilder();
        SwingUtilities.invokeLater(this::beginAssistantMessage);

        generationLatch = new CountDownLatch(1);
        CountDownLatch latch = generationLatch;

        try {
            String fullPrompt = "User: " + userPrompt + "\nAssistant:";
            String paramsJson = "{\"prompt\":\"" + fullPrompt.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r") + "\",\"n_predict\":256,\"stream\":true}";

            model.completion(paramsJson, new llama4aj.CompletionCallback() {
                @Override
                public void onTokenReceived(String token, boolean stop) {
                    if (!token.isEmpty()) {
                        currentResponse.append(token);
                        SwingUtilities.invokeLater(() -> updateStreamingMessage(currentResponse.toString()));
                    }
                    if (stop) latch.countDown();
                }
            });

        } catch (Exception e) {
            setStatus("ERROR: " + e.getMessage());
            setGeneratingState(false);
            return;
        }

        try { latch.await(); } catch (InterruptedException e) { latch.countDown(); }

        String finalText = currentResponse.toString();
        convo.saveMessage("Assistant", finalText);
        SwingUtilities.invokeLater(() -> {
            finaliseStreamingMessage(finalText);
            setGeneratingState(false);
            setStatus("Ready");
        });
    }

    private void interruptGeneration() {
        if (model != null) new Thread(() -> {
            model.interrupt();
            if (generationLatch != null) generationLatch.countDown();
        }).start();
    }

    private void newConversation() {
        if (isGenerating) return;
        convo.newConversation();
        firstMessage = true;
        chatPane.setText("");
        refreshSidebar();
        selectCurrentInSidebar();
        setStatus("New");
    }

    private void switchConversation(ConvoManager.Conversation c) {
        if (isGenerating) return;
        convo.setCurrentConversation(c.id);
        firstMessage = false;
        chatPane.setText("");
        loadHistory();
        setStatus("Loaded: " + c.title);
    }

    private void refreshSidebar() {
        List<ConvoManager.Conversation> all = convo.loadAllConversations();
        sidebarModel.clear();
        for (ConvoManager.Conversation c : all) sidebarModel.addElement(c);
        selectCurrentInSidebar();
    }

    private void selectCurrentInSidebar() {
        int current = convo.getCurrentConversationId();
        for (int i = 0; i < sidebarModel.size(); i++) {
            if (sidebarModel.get(i).id == current) {
                sidebarList.setSelectedIndex(i);
                break;
            }
        }
    }

    private void appendMessage(String author, String content) {
        StyledDocument doc = chatPane.getStyledDocument();
        try {
            SimpleAttributeSet authorStyle = new SimpleAttributeSet();
            StyleConstants.setBold(authorStyle, true);
            StyleConstants.setSpaceAbove(authorStyle, 8f);
            doc.insertString(doc.getLength(), author + ":\n", authorStyle);
            renderMarkdown(doc, content);
            doc.insertString(doc.getLength(), "\n", new SimpleAttributeSet());
            chatPane.setCaretPosition(doc.getLength());
        } catch (BadLocationException e) {
            e.printStackTrace();
        }
    }

    private void beginAssistantMessage() {
        StyledDocument doc = chatPane.getStyledDocument();
        try {
            SimpleAttributeSet authorStyle = new SimpleAttributeSet();
            StyleConstants.setBold(authorStyle, true);
            StyleConstants.setSpaceAbove(authorStyle, 8f);
            doc.insertString(doc.getLength(), "Assistant:\n", authorStyle);
            streamStart = doc.getLength();
            chatPane.setCaretPosition(doc.getLength());
        } catch (BadLocationException e) {
            e.printStackTrace();
        }
    }

    private void updateStreamingMessage(String fullText) {
        StyledDocument doc = chatPane.getStyledDocument();
        try {
            if (streamStart < 0) return;
            int tail = doc.getLength() - streamStart;
            if (tail > 0) doc.remove(streamStart, tail);
            renderMarkdown(doc, fullText);
            chatPane.setCaretPosition(doc.getLength());
        } catch (BadLocationException e) {
            e.printStackTrace();
        }
    }

    private void finaliseStreamingMessage(String fullText) {
        StyledDocument doc = chatPane.getStyledDocument();
        try {
            if (streamStart < 0) return;
            int tail = doc.getLength() - streamStart;
            if (tail > 0) doc.remove(streamStart, tail);
            renderMarkdown(doc, fullText);
            doc.insertString(doc.getLength(), "\n", new SimpleAttributeSet());
            streamStart = -1;
            chatPane.setCaretPosition(doc.getLength());
        } catch (BadLocationException e) {
            e.printStackTrace();
        }
    }

    private void renderMarkdown(StyledDocument doc, String text) throws BadLocationException {
        String[] parts = text.split("(?s)```[a-zA-Z]*\n?", -1);
        boolean inFence = false;
        for (String part : parts) {
            if (inFence) appendCode(doc, part);
            else         renderInline(doc, part);
            inFence = !inFence;
        }
    }

    private void renderInline(StyledDocument doc, String text) throws BadLocationException {
        String[] parts = text.split("`", -1);
        for (int i = 0; i < parts.length; i++) {
            if (i % 2 == 1) appendCode(doc, parts[i]);
            else             renderBold(doc, parts[i]);
        }
    }

    private void renderBold(StyledDocument doc, String text) throws BadLocationException {
        String[] parts = text.split("\\*\\*", -1);
        for (int i = 0; i < parts.length; i++) {
            SimpleAttributeSet s = new SimpleAttributeSet();
            StyleConstants.setFontFamily(s, "SansSerif");
            StyleConstants.setFontSize(s, 14);
            if (i % 2 == 1) StyleConstants.setBold(s, true);
            doc.insertString(doc.getLength(), parts[i], s);
        }
    }

    private void appendCode(StyledDocument doc, String code) throws BadLocationException {
        SimpleAttributeSet s = new SimpleAttributeSet();
        StyleConstants.setFontFamily(s, "Monospaced");
        StyleConstants.setFontSize(s, 13);
        StyleConstants.setBackground(s, darkTheme ? CODE_BG_DARK : CODE_BG_LIGHT);
        doc.insertString(doc.getLength(), code, s);
    }

    private void buildUI() {
        frame = new JFrame("Llama Chat");
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.setSize(900, 600);
        frame.setMinimumSize(new Dimension(600, 400));
        frame.addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent e) { shutdown(); }
        });

        // --- Sidebar ---
        sidebarModel = new DefaultListModel<ConvoManager.Conversation>();
        sidebarList  = new JList<ConvoManager.Conversation>(sidebarModel);
        sidebarList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        sidebarList.setFont(new Font("SansSerif", Font.PLAIN, 13));
        sidebarList.setCellRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
                super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                if (value instanceof ConvoManager.Conversation) {
                    setText(((ConvoManager.Conversation) value).title);
                }
                return this;
            }
        });
        sidebarList.addMouseListener(new MouseAdapter() {
            @Override public void mouseClicked(MouseEvent e) {
                ConvoManager.Conversation selected = sidebarList.getSelectedValue();
                if (selected != null && selected.id != convo.getCurrentConversationId()) {
                    switchConversation(selected);
                }
            }
        });
        JScrollPane sidebarScroll = new JScrollPane(sidebarList);
        sidebarScroll.setPreferredSize(new Dimension(180, 0));
        sidebarScroll.setBorder(BorderFactory.createMatteBorder(0, 0, 0, 1, Color.GRAY));

        newConvoButton = new JButton("New");
        newConvoButton.addActionListener(e -> newConversation());

        JPanel sidebarPanel = new JPanel(new BorderLayout(0, 4));
        sidebarPanel.setBorder(BorderFactory.createEmptyBorder(6, 4, 6, 4));
        sidebarPanel.add(sidebarScroll, BorderLayout.CENTER);

        // --- Chat pane ---
        chatPane = new JTextPane();
        chatPane.setEditable(false);
        chatPane.setFont(new Font("SansSerif", Font.PLAIN, 14));
        JScrollPane chatScroll = new JScrollPane(chatPane);
        chatScroll.setBorder(BorderFactory.createEmptyBorder());

        // --- Input ---
        inputArea = new JTextArea(4, 40);
        inputArea.setLineWrap(true);
        inputArea.setWrapStyleWord(true);
        inputArea.setFont(new Font("SansSerif", Font.PLAIN, 14));
        inputArea.addKeyListener(new KeyAdapter() {
            @Override public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER && !e.isShiftDown()) {
                    e.consume();
                    sendMessage();
                }
            }
        });
        JScrollPane inputScroll = new JScrollPane(inputArea);
        inputScroll.setPreferredSize(new Dimension(0, 90));
        inputScroll.setMinimumSize(new Dimension(100, 90));

        // --- Buttons ---
        sendButton  = new JButton("Send");
        stopButton  = new JButton("Stop");
        themeButton = new JButton("Dark");
        clearButton = new JButton("Clear");

        sendButton.setEnabled(false);
        stopButton.setEnabled(false);

        sendButton    .addActionListener(e -> sendMessage());
        stopButton    .addActionListener(e -> interruptGeneration());
        themeButton   .addActionListener(e -> toggleTheme());
        clearButton   .addActionListener(e -> clearHistory());

        Dimension btnSize = new Dimension(130, 32);
        sendButton    .setPreferredSize(btnSize);
        stopButton    .setPreferredSize(btnSize);
        newConvoButton.setPreferredSize(btnSize);
        themeButton   .setPreferredSize(btnSize);
        clearButton   .setPreferredSize(btnSize);

        JPanel buttonPanel = new JPanel(new GridLayout(5, 1, 0, 4));
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(0, 6, 0, 0));
        buttonPanel.add(sendButton);
        buttonPanel.add(stopButton);
        buttonPanel.add(newConvoButton);
        buttonPanel.add(themeButton);
        buttonPanel.add(clearButton);

        statusLabel = new JLabel("Starting...");
        statusLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        statusLabel.setBorder(BorderFactory.createEmptyBorder(3, 2, 2, 2));

        JPanel inputRow = new JPanel(new BorderLayout(0, 0));
        inputRow.add(inputScroll, BorderLayout.CENTER);
        inputRow.add(buttonPanel, BorderLayout.EAST);

        JPanel bottomPanel = new JPanel(new BorderLayout(0, 4));
        bottomPanel.setBorder(BorderFactory.createEmptyBorder(6, 8, 8, 8));
        bottomPanel.add(inputRow,    BorderLayout.CENTER);
        bottomPanel.add(statusLabel, BorderLayout.SOUTH);

        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(chatScroll,  BorderLayout.CENTER);
        mainPanel.add(bottomPanel, BorderLayout.SOUTH);

        frame.setLayout(new BorderLayout());
        frame.add(sidebarPanel, BorderLayout.WEST);
        frame.add(mainPanel,    BorderLayout.CENTER);
        frame.setVisible(true);
    }

    private void toggleTheme() {
        darkTheme = !darkTheme;
        try {
            if (darkTheme) { FlatDarkLaf.setup();  themeButton.setText("Light"); }
            else           { FlatLightLaf.setup(); themeButton.setText("Dark");  }
            SwingUtilities.updateComponentTreeUI(frame);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void loadHistory() {
        List<ConvoManager.Message> history = convo.loadHistory();
        for (ConvoManager.Message msg : history) {
            appendMessage(msg.author, msg.content);
        }
        if (!history.isEmpty()) {
            firstMessage = false;
            setStatus("History loaded (" + history.size() + " messages)");
        }
    }

    private void clearHistory() {
        int ok = JOptionPane.showConfirmDialog(frame,
            "Clear current conversation?", "Clear", JOptionPane.YES_NO_OPTION);
        if (ok == JOptionPane.YES_OPTION) {
            convo.clearHistory();
            chatPane.setText("");
            firstMessage = true;
            setStatus("Cleared");
        }
    }

    private void setGeneratingState(boolean generating) {
        isGenerating = generating;
        SwingUtilities.invokeLater(() -> {
            sendButton    .setEnabled(!generating);
            stopButton    .setEnabled(generating);
            newConvoButton.setEnabled(!generating);
            inputArea     .setEnabled(!generating);
            if (generating) setStatus("Generating...");
        });
    }

    private void setStatus(String text)     { SwingUtilities.invokeLater(() -> statusLabel.setText(text)); }
    private void setSendEnabled(boolean on) { SwingUtilities.invokeLater(() -> sendButton.setEnabled(on)); }

    private void shutdown() {
        if (executor != null && !executor.isShutdown()) executor.shutdownNow();
        if (model    != null)                           model.close();
        System.exit(0);
    }
}
