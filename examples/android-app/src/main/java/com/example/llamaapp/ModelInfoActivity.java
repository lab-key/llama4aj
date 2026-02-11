package com.example.llamaapp;

import com.example.llamaapp.ModelManager;
import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class ModelInfoActivity extends AppCompatActivity {

    private static final String TAG = "ModelInfoActivity";
    private EditText systemPromptInput, maxTokensInput, temperatureInput;
    private TextView modelPathText, modelInfoText;
    private Button setModelButton, getModelButton; // Removed saveConfigButton

    private ModelManager modelManager;
    private ActivityResultLauncher<Intent> pickModelLauncher;
    private Toolbar toolbar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_info);

        toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setTitle("Model Settings");
            getSupportActionBar().setDisplayHomeAsUpEnabled(true); // Enable back button
        }

        modelManager = ModelManager.getInstance(this);

        systemPromptInput = findViewById(R.id.system_prompt_input);
        maxTokensInput = findViewById(R.id.max_tokens_input);
        temperatureInput = findViewById(R.id.temperature_input);
        modelPathText = findViewById(R.id.model_path_text);
        modelInfoText = findViewById(R.id.model_info_text);
        setModelButton = findViewById(R.id.set_model_button);
        getModelButton = findViewById(R.id.get_model_button);

        loadConfigToUI();

        pickModelLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK && result.getData() != null) {
                    Uri uri = result.getData().getData();
                    if (uri != null) {
                        handlePickedModelUri(uri);
                    }
                }
            });

        setModelButton.setOnClickListener(v -> pickModelFile());
        getModelButton.setOnClickListener(v -> {
            Intent intent = new Intent(ModelInfoActivity.this, ModelDownloadActivity.class);
            startActivity(intent);
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.model_info_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
 if (item.getItemId() == android.R.id.home) {
            // Handle the back button in the toolbar
            onBackPressed();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void loadConfigToUI() {
        systemPromptInput.setText(modelManager.getSystemPrompt());
        maxTokensInput.setText(String.valueOf(modelManager.getMaxTokens()));
        temperatureInput.setText(String.valueOf(modelManager.getTemperature()));
        modelPathText.setText("Selected: " + modelManager.getModelPath());
        // modelInfoText.setText(modelManager.getModelInfo()); // ModelManager.java doesn't have getModelInfo yet
    }

    private void saveConfigFromUI() {
        String systemPrompt = systemPromptInput.getText().toString();
        int maxTokens = Integer.parseInt(maxTokensInput.getText().toString());
        float temperature = Float.parseFloat(temperatureInput.getText().toString());

        modelManager.saveConfig(modelManager.getModelPath(), systemPrompt, maxTokens, temperature);
        Toast.makeText(this, "Configuration saved!", Toast.LENGTH_SHORT).show();
    }

    private void pickModelFile() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("*/*"); // Allow all file types
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        pickModelLauncher.launch(intent);
    }

    private void handlePickedModelUri(Uri uri) {
        Toast.makeText(this, "Processing model file...", Toast.LENGTH_SHORT).show();
        
        java.util.concurrent.Executors.newSingleThreadExecutor().execute(() -> {
            try {
                String modelPath = copyUriToFile(uri);
                runOnUiThread(() -> {
                    if (modelPath != null) {
                        if (modelManager.validateModelFile(modelPath)) {
                            modelManager.saveConfig(modelPath, modelManager.getSystemPrompt(), modelManager.getMaxTokens(), modelManager.getTemperature());
                            loadConfigToUI();
                            Toast.makeText(ModelInfoActivity.this, "Model selected and saved!", Toast.LENGTH_SHORT).show();
                        } else {
                            Toast.makeText(ModelInfoActivity.this, "Invalid model file selected.", Toast.LENGTH_LONG).show();
                        }
                    } else {
                        Toast.makeText(ModelInfoActivity.this, "Could not copy model file.", Toast.LENGTH_LONG).show();
                    }
                });
            } catch (Exception e) {
                Log.e(TAG, "Error handling picked model URI", e);
                runOnUiThread(() -> Toast.makeText(ModelInfoActivity.this, "Error: " + e.getMessage(), Toast.LENGTH_LONG).show());
            }
        });
    }

    private String copyUriToFile(Uri uri) throws IOException {
        File cacheDir = getApplicationContext().getCacheDir();
        String fileName = getFileName(uri);
        if (fileName == null) {
            fileName = "model_file_" + System.currentTimeMillis();
        }
        File tempFile = new File(cacheDir, fileName);

        try (InputStream inputStream = getContentResolver().openInputStream(uri);
             FileOutputStream outputStream = new FileOutputStream(tempFile)) {
            if (inputStream == null) {
                throw new IOException("Unable to open input stream for URI: " + uri);
            }
            byte[] buffer = new byte[4 * 1024]; // 4KB buffer
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            outputStream.flush();
        }
        return tempFile.getAbsolutePath();
    }

    private String getFileName(Uri uri) {
        String result = null;
        if (uri.getScheme().equals("content")) {
            try (android.database.Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                    if (nameIndex != -1) {
                        result = cursor.getString(nameIndex);
                    }
                }
            }
        }
        if (result == null) {
            result = uri.getLastPathSegment();
        }
        return result;
    }
}
