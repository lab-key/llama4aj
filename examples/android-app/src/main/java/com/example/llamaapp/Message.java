package com.example.llamaapp;

public class Message {
    private String text;
    private boolean isUser; // true for user message, false for model response

    public Message(String text, boolean isUser) {
        this.text = text;
        this.isUser = isUser;
    }

    public String getText() {
        return text;
    }

    public boolean isUser() {
        return isUser;
    }
}
