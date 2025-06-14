import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, X, Image, FileText, Command } from 'lucide-react';
import { useChatStore, Attachment } from '../../store/chatStore';
import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';

export const ChatInput: React.FC = () => {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [showCommands, setShowCommands] = useState(false);
  const { activeChat, addMessage, setLoading, isLoading } = useChatStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const commands = [
    { name: '/imagine', description: 'Generate an image' },
    { name: '/reset', description: 'Clear conversation' },
    { name: '/summarize', description: 'Summarize conversation' },
    { name: '/explain', description: 'Explain in detail' },
  ];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '/') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (message.startsWith('/')) {
      setShowCommands(true);
    } else {
      setShowCommands(false);
    }
  }, [message]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim() || !activeChat || isLoading) return;

    const userMessage = {
      content: message.trim(),
      role: 'user' as const,
      attachments: attachments.length > 0 ? attachments : undefined,
    };

    addMessage(activeChat, userMessage);
    setMessage('');
    setAttachments([]);

    setLoading(true);
    
    setTimeout(() => {
      const responses = [
        "I understand your question. Let me help you with that...",
        "That's an interesting point! Here's what I think...",
        "Great question! Based on my knowledge, I can tell you that...",
        "I'd be happy to help with that. Let me break it down for you...",
        "Thanks for asking! Here's a comprehensive answer...",
      ];
      
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      
      addMessage(activeChat, {
        content: randomResponse + "\n\n(This is a simulated response. In the full implementation, this would connect to your chosen AI provider.)",
        role: 'assistant',
      });
      
      setLoading(false);
    }, 1500 + Math.random() * 1000);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    
    files.forEach(file => {
      const attachment: Attachment = {
        id: Date.now().toString() + Math.random(),
        name: file.name,
        type: file.type,
        size: file.size,
        url: URL.createObjectURL(file),
      };
      
      setAttachments(prev => [...prev, attachment]);
    });

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(prev => prev.filter(att => att.id !== id));
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return <Image className="w-4 h-4" />;
    return <FileText className="w-4 h-4" />;
  };

  const handleCommandSelect = (command: string) => {
    setMessage(command + ' ');
    setShowCommands(false);
    textareaRef.current?.focus();
  };

  const filteredCommands = commands.filter(cmd => 
    cmd.name.toLowerCase().includes(message.toLowerCase())
  );

  return (
    <div className="border-t border-border bg-card/50 backdrop-blur-xl">
      {/* Commands Dropdown */}
      {showCommands && filteredCommands.length > 0 && (
        <div className="p-2 border-b border-border">
          <div className="bg-background/80 rounded-xl border border-border shadow-lg">
            {filteredCommands.map((command) => (
              <button
                key={command.name}
                onClick={() => handleCommandSelect(command.name)}
                className="w-full p-3 text-left hover:bg-muted/50 first:rounded-t-xl last:rounded-b-xl transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <Command className="w-4 h-4 text-primary" />
                  <div>
                    <div className="font-medium text-sm">{command.name}</div>
                    <div className="text-xs text-muted-foreground">{command.description}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Attachments Preview */}
      {attachments.length > 0 && (
        <div className="p-4 border-b border-border">
          <div className="flex flex-wrap gap-2">
            {attachments.map((attachment) => (
              <div key={attachment.id} className="flex items-center space-x-2 bg-background/50 rounded-lg p-2 pr-1 border border-border">
                {getFileIcon(attachment.type)}
                <span className="text-sm truncate max-w-32">{attachment.name}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeAttachment(attachment.id)}
                  className="p-1 h-auto text-muted-foreground hover:text-destructive"
                >
                  <X className="w-3 h-3" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="p-4">
        <div className="flex items-end space-x-3 bg-background/50 rounded-2xl border border-border p-3 shadow-lg">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            className="flex-shrink-0 p-2 hover:bg-muted/50 rounded-xl"
          >
            <Paperclip className="w-4 h-4" />
          </Button>

          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything..."
              className="min-h-[44px] max-h-32 resize-none bg-transparent border-0 focus:ring-0 focus:border-0 text-sm placeholder:text-muted-foreground/70"
              disabled={isLoading}
            />
          </div>

          <Button 
            type="submit" 
            disabled={!message.trim() || isLoading}
            className="flex-shrink-0 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl shadow-lg hover:shadow-xl transition-all duration-200"
            size="sm"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>

        <div className="mt-2 text-xs text-muted-foreground px-3">
          Press <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs font-mono">Enter</kbd> to send, 
          <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs font-mono ml-1">Shift+Enter</kbd> for new line,
          type <kbd className="px-1.5 py-0.5 bg-muted rounded text-xs font-mono ml-1">/</kbd> for commands
        </div>
      </form>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        accept="image/*,.pdf,.doc,.docx,.txt"
        onChange={handleFileSelect}
      />
    </div>
  );
};
