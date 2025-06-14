
import React from 'react';
import { User, Bot, Copy, Check, FileText, ExternalLink } from 'lucide-react';
import { Message } from '../../store/chatStore';
import { Button } from '../ui/button';
import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const [copied, setCopied] = useState(false);
  const [previewAttachment, setPreviewAttachment] = useState<any>(null);
  const [showTimestamp, setShowTimestamp] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatTime = (timestamp: Date | string) => {
    const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
    if (isNaN(date.getTime())) {
      return '--:--';
    }
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const openAttachmentPreview = (attachment: any) => {
    if (attachment.type.startsWith('image/')) {
      setPreviewAttachment(attachment);
    } else {
      window.open(attachment.url, '_blank');
    }
  };

  const renderAttachmentThumbnail = (attachment: any) => {
    if (attachment.type.startsWith('image/')) {
      return (
        <div className="relative group cursor-pointer" onClick={() => openAttachmentPreview(attachment)}>
          <img
            src={attachment.url}
            alt={attachment.name}
            className="w-32 h-32 object-cover rounded-xl border border-border hover:opacity-90 transition-opacity shadow-md"
          />
          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 rounded-xl transition-colors flex items-center justify-center">
            <ExternalLink className="w-4 h-4 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <div className="mt-2 text-xs text-muted-foreground truncate max-w-32">
            {attachment.name}
          </div>
        </div>
      );
    } else {
      return (
        <div 
          className="flex items-center space-x-3 p-3 bg-background/50 rounded-xl border border-border cursor-pointer hover:bg-background/70 transition-colors max-w-64 shadow-sm"
          onClick={() => openAttachmentPreview(attachment)}
        >
          <FileText className="w-8 h-8 text-muted-foreground flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium truncate">{attachment.name}</div>
            <div className="text-xs text-muted-foreground">
              {(attachment.size / 1024 / 1024).toFixed(1)} MB
            </div>
          </div>
          <ExternalLink className="w-4 h-4 text-muted-foreground flex-shrink-0" />
        </div>
      );
    }
  };

  const getUserInitials = () => {
    if (isUser) {
      return 'U';
    }
    return 'AI';
  };

  return (
    <>
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6 message-animate`}>
        <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-3 max-w-4xl group`}>
          {/* Profile Avatar */}
          <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold ${
            isUser ? 'bg-primary text-primary-foreground ml-3' : 'bg-muted text-muted-foreground mr-3'
          }`}>
            {getUserInitials()}
          </div>

          {/* Message Content */}
          <div 
            className={`relative ${isUser ? 'message-user' : 'message-ai'} p-4 max-w-[80%] transition-all duration-200 hover:shadow-xl`}
            onMouseEnter={() => setShowTimestamp(true)}
            onMouseLeave={() => setShowTimestamp(false)}
          >
            {/* Text Content */}
            {message.content && (
              <div className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                {message.content}
              </div>
            )}
            
            {/* Attachments */}
            {message.attachments && message.attachments.length > 0 && (
              <div className={`${message.content ? 'mt-3' : ''} space-y-3`}>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {message.attachments.map((attachment) => (
                    <div key={attachment.id}>
                      {renderAttachmentThumbnail(attachment)}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Message Actions */}
            <div className={`flex items-center justify-between mt-3 pt-2 border-t ${
              isUser ? 'border-white/20' : 'border-border'
            }`}>
              <div className="flex items-center space-x-2">
                <span className={`text-xs transition-opacity ${
                  showTimestamp ? 'opacity-100' : 'opacity-0'
                }`}>
                  {formatTime(message.timestamp)}
                </span>
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-1 h-auto hover:bg-white/10"
              >
                {copied ? (
                  <Check className="w-3 h-3" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Image Preview Modal */}
      <Dialog open={!!previewAttachment} onOpenChange={() => setPreviewAttachment(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden bg-card border-border">
          <DialogHeader>
            <DialogTitle>{previewAttachment?.name}</DialogTitle>
          </DialogHeader>
          <div className="flex items-center justify-center p-4">
            {previewAttachment && (
              <img
                src={previewAttachment.url}
                alt={previewAttachment.name}
                className="max-w-full max-h-[70vh] object-contain rounded-lg"
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};
