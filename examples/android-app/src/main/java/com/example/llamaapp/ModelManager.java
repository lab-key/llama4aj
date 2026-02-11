package com.example.llamaapp;

import android.content.SharedPreferences;
import android.util.Log;

import java.io.File;

import com.llama4aj.LlamaContext;

public class ModelManager {
    private static final String TAG = "ModelManager";
    private static final String PREFS_NAME = "LlamaModelConfig";
    private static final String KEY_MODEL_PATH = "modelPath";
    private static final String KEY_SYSTEM_PROMPT = "systemPrompt";
    private static final String KEY_MAX_TOKENS = "maxTokens";
    private static final String KEY_TEMPERATURE = "temperature";

    private static volatile ModelManager instance;
    private final SharedPreferences preferences;

    private volatile LlamaContext llamaContext;
    private volatile String loadedModelPath;

    private String currentSystemPrompt;
    private int currentMaxTokens;
    private float currentTemperature;

    private ModelManager(android.content.Context context) {
        preferences = context.getSharedPreferences(PREFS_NAME, android.content.Context.MODE_PRIVATE);
        loadConfig();
    }

    public static ModelManager getInstance(android.content.Context context) {
        if (instance == null) {
            synchronized (ModelManager.class) {
                if (instance == null) {
                    instance = new ModelManager(context.getApplicationContext());
                }
            }
        }
        return instance;
    }

    private void loadConfig() {
        currentSystemPrompt = preferences.getString(KEY_SYSTEM_PROMPT, "You are a helpful AI assistant.");
        currentMaxTokens = preferences.getInt(KEY_MAX_TOKENS, 400);
        currentTemperature = preferences.getFloat(KEY_TEMPERATURE, 0.7f);
    }

    public void saveConfig(String modelPath, String systemPrompt, int maxTokens, float temperature) {
        SharedPreferences.Editor editor = preferences.edit();
        editor.putString(KEY_MODEL_PATH, modelPath);
        editor.putString(KEY_SYSTEM_PROMPT, systemPrompt);
        editor.putInt(KEY_MAX_TOKENS, maxTokens);
        editor.putFloat(KEY_TEMPERATURE, temperature);
        editor.apply();

        // Reload current values
        currentSystemPrompt = systemPrompt;
        currentMaxTokens = maxTokens;
        currentTemperature = temperature;
    }

    public String getModelPath() {
        return preferences.getString(KEY_MODEL_PATH, "");
    }

    public String getSystemPrompt() {
        return currentSystemPrompt;
    }

    public int getMaxTokens() {
        return currentMaxTokens;
    }

    public float getTemperature() {
        return currentTemperature;
    }

    public synchronized LlamaContext getLlamaContext() {
        return llamaContext;
    }

    public synchronized boolean isModelLoaded() {
        return llamaContext != null && loadedModelPath != null;
    }

    public synchronized String getLoadedModelPath() {
        return loadedModelPath;
    }

    public synchronized boolean loadModel(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            Log.e(TAG, "Model path is empty");
            return false;
        }

        File modelFile = new File(modelPath);
        if (!modelFile.exists()) {
            Log.e(TAG, "Model file does not exist: " + modelPath);
            return false;
        }

        // If same model is already loaded, don't reload
        if (llamaContext != null && modelPath.equals(loadedModelPath)) {
            Log.d(TAG, "Model already loaded");
            return true;
        }

        // Unload existing model first
        if (llamaContext != null) {
            Log.d(TAG, "Unloading existing model");
            try {
                llamaContext.destroy();
            } catch (Exception e) {
                Log.w(TAG, "Error destroying previous context", e);
            }
            llamaContext = null;
            loadedModelPath = null;
        }

        // Load new model
        try {
            Log.d(TAG, "Loading model from: " + modelPath);
            LlamaContext newContext = LlamaContext.create(modelPath);

            if (newContext == null) {
                Log.e(TAG, "LlamaContext.create returned null");
                return false;
            }

            llamaContext = newContext;
            loadedModelPath = modelPath;
            Log.d(TAG, "Model loaded successfully");
            return true;

        } catch (Exception e) {
            Log.e(TAG, "Error loading model", e);
            llamaContext = null;
            loadedModelPath = null;
            return false;
        }
    }

    public synchronized void unloadModel() {
        if (llamaContext != null) {
            try {
                llamaContext.destroy();
                Log.d(TAG, "Model unloaded");
            } catch (Exception e) {
                Log.e(TAG, "Error unloading model", e);
            }
            llamaContext = null;
            loadedModelPath = null;
        }
    }

    public boolean validateModelFile(String path) {
        if (path == null || path.isEmpty()) {
            return false;
        }
        File file = new File(path);
        return file.exists() && file.isFile() && file.length() > 0;
    }
}
