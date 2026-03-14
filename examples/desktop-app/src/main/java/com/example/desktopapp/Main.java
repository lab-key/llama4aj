package com.example.desktopapp;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import com.llama4aj;
import org.json.JSONObject;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main extends Application implements llama4aj.CompletionCallback {

    private TextArea chatArea;
    private TextField promptInput;
    private Button sendButton;
    private Button stopButton;

    private llama4aj model;
    private ExecutorService executorService;
    private StringBuilder currentResponse;
    private volatile boolean isGenerating = false;

    // Placeholder for model path - user needs to set this
    private static final String MODEL_PATH = "./model.gguf"; // !!! IMPORTANT: CHANGE THIS !!!

    @Override
    public void start(Stage primaryStage) {
        chatArea = new TextArea();
        chatArea.setEditable(false);
        chatArea.setWrapText(true);

        promptInput = new TextField();
        promptInput.setPromptText("Enter your prompt...");

        sendButton = new Button("Send");
        sendButton.setOnAction(e -> sendMessage());

        stopButton = new Button("Stop");
        stopButton.setDisable(true);
        stopButton.setOnAction(e -> interruptGeneration());

        VBox root = new VBox(10, chatArea, promptInput, sendButton, stopButton);
        Scene scene = new Scene(root, 600, 400);

        primaryStage.setTitle("Llama Desktop App");
        primaryStage.setScene(scene);
        primaryStage.show();

        executorService = Executors.newSingleThreadExecutor();
        currentResponse = new StringBuilder();

        // Initialize model in a background thread
        executorService.submit(this::initializeModel);
    }

    private void initializeModel() {
        Platform.runLater(() -> chatArea.appendText("Loading model...\n"));
        File modelFile = new File(MODEL_PATH);
        if (!modelFile.exists()) {
            Platform.runLater(() -> chatArea.appendText("ERROR: Model file not found at " + MODEL_PATH + "\n"));
            Platform.runLater(() -> sendButton.setDisable(true));
            return;
        }

        try {
            model = llama4aj.load(MODEL_PATH);
            if (model != null) {
                Platform.runLater(() -> chatArea.appendText("Model loaded successfully!\n"));
                Platform.runLater(() -> sendButton.setDisable(false));
            } else {
                Platform.runLater(() -> chatArea.appendText("ERROR: Failed to load model.\n"));
                Platform.runLater(() -> sendButton.setDisable(true));
            }
        } catch (Exception e) {
            Platform.runLater(() -> chatArea.appendText("ERROR during model loading: " + e.getMessage() + "\n"));
            Platform.runLater(() -> sendButton.setDisable(true));
        }
    }

    private void sendMessage() {
        String prompt = promptInput.getText().trim();
        if (prompt.isEmpty() || isGenerating) {
            return;
        }

        chatArea.appendText("You: " + prompt + "\n");
        promptInput.clear();
        setGeneratingState(true);

        executorService.submit(() -> generateResponse(prompt));
    }

    private void generateResponse(String userPrompt) {
        if (model == null) {
            Platform.runLater(() -> chatArea.appendText("ERROR: Model not loaded.\n"));
            setGeneratingState(false);
            return;
        }

        try {
            // Simple prompt building for now
            String fullPrompt = "User: " + userPrompt + "\nAssistant:";
            JSONObject params = new JSONObject();
            params.put("prompt", fullPrompt);
            params.put("n_predict", 256); // Max tokens to generate
            params.put("temperature", 0.7);
            params.put("top_k", 40);
            params.put("top_p", 0.9);
            params.put("repeat_penalty", 1.1);
            params.put("stream", true);

            currentResponse = new StringBuilder();
            Platform.runLater(() -> chatArea.appendText("Assistant: "));
            model.completion(params.toString(), this);

        } catch (Exception e) {
            Platform.runLater(() -> chatArea.appendText("ERROR during generation: " + e.getMessage() + "\n"));
            setGeneratingState(false);
        }
    }

    @Override
    public void onTokenReceived(String tokenDataJson) {
        try {
            JSONObject json = new JSONObject(tokenDataJson);
            String content = json.optString("content", "");
            boolean stop = json.optBoolean("stop", false);

            currentResponse.append(content);
            Platform.runLater(() -> {
                // Update the last line of chatArea
                String text = chatArea.getText();
                int lastNewline = text.lastIndexOf('\n');
                if (lastNewline != -1 && text.substring(lastNewline).startsWith("\nAssistant: ")) {
                    chatArea.replaceText(lastNewline + 1, text.length(), "Assistant: " + currentResponse.toString());
                } else {
                    chatArea.appendText(currentResponse.toString());
                }
            });

            if (stop) {
                Platform.runLater(() -> chatArea.appendText("\n"));
                setGeneratingState(false);
            }

        } catch (Exception e) {
            Platform.runLater(() -> chatArea.appendText("ERROR parsing token: " + e.getMessage() + "\n"));
            setGeneratingState(false);
        }
    }

    private void interruptGeneration() {
        if (model != null) {
            executorService.submit(() -> model.interrupt());
        }
    }

    private void setGeneratingState(boolean generating) {
        isGenerating = generating;
        Platform.runLater(() -> {
            sendButton.setDisable(generating);
            stopButton.setDisable(!generating);
            promptInput.setDisable(generating);
        });
    }

    @Override
    public void stop() {
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdownNow();
        }
        if (model != null) {
            model.close();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
