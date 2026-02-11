package com.example.llamaapp;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.MessageViewHolder> {

    private List<Message> messageList;

    public ChatAdapter(List<Message> messageList) {
        this.messageList = messageList;
    }

    @NonNull
    @Override
    public MessageViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_chat_message, parent, false);
        return new MessageViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull MessageViewHolder holder, int position) {
        Message message = messageList.get(position);
        holder.messageTextView.setText(message.getText());

        // Adjust layout parameters based on whether it's a user or model message
        android.widget.LinearLayout.LayoutParams layoutParams = (android.widget.LinearLayout.LayoutParams) holder.messageTextView.getLayoutParams();
        if (message.isUser()) {
            holder.messageTextView.setBackgroundResource(R.drawable.message_background_user);
            holder.messageTextView.setTextColor(android.graphics.Color.WHITE);
            layoutParams.gravity = android.view.Gravity.END;
            layoutParams.setMarginStart(dpToPx(holder.itemView, 60));
            layoutParams.setMarginEnd(dpToPx(holder.itemView, 0));
        } else {
            holder.messageTextView.setBackgroundResource(R.drawable.message_background_model);
            holder.messageTextView.setTextColor(android.graphics.Color.BLACK);
            layoutParams.gravity = android.view.Gravity.START;
            layoutParams.setMarginStart(dpToPx(holder.itemView, 0));
            layoutParams.setMarginEnd(dpToPx(holder.itemView, 60));
        }
        holder.messageTextView.setLayoutParams(layoutParams);
    }

    @Override
    public int getItemCount() {
        return messageList.size();
    }

    public void addMessage(Message message) {
        messageList.add(message);
        notifyItemInserted(messageList.size() - 1);
    }

    static class MessageViewHolder extends RecyclerView.ViewHolder {
        TextView messageTextView;

        public MessageViewHolder(@NonNull View itemView) {
            super(itemView);
            messageTextView = itemView.findViewById(R.id.message_text_view);
        }
    }

    private int dpToPx(View view, int dp) {
        float density = view.getContext().getResources().getDisplayMetrics().density;
        return Math.round((float) dp * density);
    }
}
