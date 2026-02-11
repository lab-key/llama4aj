package com.example.llamaapp;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.llama4aj.LlamaContext;

public class MainActivity extends AppCompatActivity implements LlamaContext.CompletionCallback {

    private static final String TAG = "LlamaApp";

    private RecyclerView chatRecyclerView;
    private EditText promptInput;
    private ImageButton sendButton;
    private ImageButton stopButton;

    private ChatAdapter chatAdapter;
    private List<Message> messageList;
    private ModelManager modelManager;
    private ExecutorService executorService;
    private Handler mainHandler;

    private volatile boolean isGenerating = false;
    private StringBuilder currentResponse;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setTitle("LlamaApp");
        }

        chatRecyclerView = findViewById(R.id.chat_recycler_view);
        promptInput = findViewById(R.id.prompt_input);
        sendButton = findViewById(R.id.send_button);
        stopButton = findViewById(R.id.stop_button);

        messageList = new ArrayList<>();
        chatAdapter = new ChatAdapter(messageList);
        chatRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        chatRecyclerView.setAdapter(chatAdapter);

        modelManager = ModelManager.getInstance(this);
        executorService = Executors.newSingleThreadExecutor();
        mainHandler = new Handler(Looper.getMainLooper());

        sendButton.setOnClickListener(v -> {
            String prompt = promptInput.getText().toString().trim();
            if (!prompt.isEmpty() && !isGenerating) {
                sendMessage(prompt);
            }
        });

        stopButton.setOnClickListener(v -> {
            if (isGenerating) {
                executorService.execute(() -> {
                    LlamaContext context = modelManager.getLlamaContext();
                    if (context != null) {
                        context.interrupt();
                    }
                });
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        checkAndLoadModel();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == R.id.action_settings) {
            Intent intent = new Intent(MainActivity.this, ModelInfoActivity.class);
            startActivity(intent);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void checkAndLoadModel() {
        String modelPath = modelManager.getModelPath();

        if (modelPath == null || modelPath.isEmpty()) {
            Toast.makeText(this, "Please select a model from settings", Toast.LENGTH_LONG).show();
            sendButton.setEnabled(false);
            return;
        }

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            Toast.makeText(this, "Model file not found: " + modelFile.getName(), Toast.LENGTH_LONG).show();
            sendButton.setEnabled(false);
            return;
        }

        // Check if model is already loaded with the same path
        if (modelManager.isModelLoaded() && modelPath.equals(modelManager.getLoadedModelPath())) {
            Log.d(TAG, "Model already loaded");
            sendButton.setEnabled(true);
            return;
        }

        // Load the model
        sendButton.setEnabled(false);
        addSystemMessage("Loading model: " + modelFile.getName() + "...");

        executorService.execute(() -> {
            boolean success = modelManager.loadModel(modelPath);
            mainHandler.post(() -> {
                if (success) {
                    addSystemMessage("Model loaded successfully!");
                    sendButton.setEnabled(true);
                } else {
                    addSystemMessage("Failed to load model. Please check the file.");
                    sendButton.setEnabled(false);
                }
            });
        });
    }

    private void sendMessage(String userMessage) {
        addUserMessage(userMessage);
        promptInput.setText("");
        generateResponse(userMessage);
    }

    private void generateResponse(String userMessage) {
        isGenerating = true;
        sendButton.setEnabled(false);
        sendButton.setVisibility(View.GONE);
        stopButton.setVisibility(View.VISIBLE);
        stopButton.setEnabled(true);
        promptInput.setEnabled(false);
        currentResponse = new StringBuilder();

        // Add empty model message placeholder
        addModelMessage("");

        executorService.execute(() -> {
            LlamaContext context = modelManager.getLlamaContext();

            if (context == null) {
                mainHandler.post(() -> {
                    updateLastModelMessage("Error: Model not loaded");
                    resetInputState();
                });
                return;
            }

            try {
                String fullPrompt = buildPrompt(userMessage);
                JSONObject params = new JSONObject();
                params.put("prompt", fullPrompt);
                params.put("n_predict", modelManager.getMaxTokens());
                params.put("temperature", modelManager.getTemperature());
                params.put("top_k", 40);
                params.put("top_p", 0.9);
                params.put("repeat_penalty", 1.1);

                Log.d(TAG, "Starting completion with prompt: " + fullPrompt);
                context.completion(params.toString(), this);

            } catch (JSONException e) {
                Log.e(TAG, "JSON error", e);
                mainHandler.post(() -> {
                    updateLastModelMessage("Error: Failed to create request");
                    resetInputState();
                });
            } catch (Exception e) {
                Log.e(TAG, "Generation error", e);
                mainHandler.post(() -> {
                    updateLastModelMessage("Error: " + e.getMessage());
                    resetInputState();
                });
            }
        });
    }

    private String buildPrompt(String userMessage) {
        String systemPrompt = modelManager.getSystemPrompt();
        if (systemPrompt != null && !systemPrompt.trim().isEmpty()) {
            return systemPrompt.trim() + "\n\nUser: " + userMessage + "\n\nAssistant:";
        }
        return userMessage;
    }

    @Override
    public void onTokenReceived(String tokenDataJson) {
        try {
            JSONObject json = new JSONObject(tokenDataJson);
            String content = json.optString("content", "");
            boolean stop = json.optBoolean("stop", false);

            currentResponse.append(content);
            final String fullResponse = currentResponse.toString();

            mainHandler.post(() -> {
                updateLastModelMessage(fullResponse);
                chatRecyclerView.scrollToPosition(messageList.size() - 1);
            });

            if (stop) {
                mainHandler.post(this::resetInputState);
            }

        } catch (JSONException e) {
            Log.e(TAG, "Error parsing token JSON", e);
            mainHandler.post(() -> {
                if (currentResponse.length() == 0) {
                    updateLastModelMessage("Error: Invalid response from model");
                }
                resetInputState();
            });
        }
    }

    private void addUserMessage(String text) {
        mainHandler.post(() -> {
            messageList.add(new Message(text, true));
            chatAdapter.notifyItemInserted(messageList.size() - 1);
            chatRecyclerView.scrollToPosition(messageList.size() - 1);
        });
    }

    private void addModelMessage(String text) {
        mainHandler.post(() -> {
            messageList.add(new Message(text, false));
            chatAdapter.notifyItemInserted(messageList.size() - 1);
            chatRecyclerView.scrollToPosition(messageList.size() - 1);
        });
    }

    private void addSystemMessage(String text) {
        addModelMessage(text);
    }

    private void updateLastModelMessage(String text) {
        if (!messageList.isEmpty()) {
            Message lastMsg = messageList.get(messageList.size() - 1);
            if (!lastMsg.isUser()) {
                messageList.set(messageList.size() - 1, new Message(text, false));
                chatAdapter.notifyItemChanged(messageList.size() - 1);
            }
        }
    }

    private void resetInputState() {
        isGenerating = false;
        sendButton.setEnabled(true);
        sendButton.setVisibility(View.VISIBLE);
        stopButton.setVisibility(View.GONE);
        promptInput.setEnabled(true);
        currentResponse = null;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (executorService != null && !executorService.isShutdown()) {
            executorService.shutdown();
        }
    }
}
