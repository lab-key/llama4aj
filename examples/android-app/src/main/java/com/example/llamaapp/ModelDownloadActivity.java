package com.example.llamaapp;

import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import androidx.appcompat.app.AppCompatActivity;

public class ModelDownloadActivity extends AppCompatActivity {

    private static final String HUGGING_FACE_URL = "https://huggingface.co/models?sort=trending&search=llama%20gguf";
    private WebView webView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_download);

        webView = findViewById(R.id.webview);

        WebSettings webSettings = webView.getSettings();
        webSettings.setJavaScriptEnabled(true); // Enable JavaScript
        webSettings.setDomStorageEnabled(true); // Enable DOM storage

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                // Handle URL loading here. For now, just load the URL in the WebView.
                // In a more advanced implementation, you might intercept download links
                // and handle them with Android's DownloadManager.
                view.loadUrl(url);
                return true;
            }
        });

        // Load the Hugging Face URL
        webView.loadUrl(HUGGING_FACE_URL);
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}
