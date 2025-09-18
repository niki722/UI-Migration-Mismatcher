import React, { useState, useCallback } from 'react';
import { Upload, X, Image, CheckCircle } from 'lucide-react';
import { Button } from './button';
import { Card, CardContent } from './card';
import { cn } from '@/lib/utils';

interface FileUploadProps {
  title: string;
  description: string;
  files: File[];
  onFilesChange: (files: File[]) => void;
  accept?: string;
  maxFiles?: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  title,
  description,
  files,
  onFilesChange,
  accept = "image/*",
  maxFiles = 10
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files).filter(file => 
      file.type.startsWith('image/')
    );
    
    if (droppedFiles.length > 0) {
      const newFiles = [...files, ...droppedFiles].slice(0, maxFiles);
      onFilesChange(newFiles);
    }
  }, [files, onFilesChange, maxFiles]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const newFiles = [...files, ...selectedFiles].slice(0, maxFiles);
    onFilesChange(newFiles);
  }, [files, onFilesChange, maxFiles]);

  const removeFile = useCallback((index: number) => {
    const newFiles = files.filter((_, i) => i !== index);
    onFilesChange(newFiles);
  }, [files, onFilesChange]);

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
          
          <div
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center transition-all",
              isDragOver 
                ? "border-primary bg-primary/5 scale-105" 
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-4">
              <Upload className={cn(
                "w-12 h-12 transition-colors",
                isDragOver ? "text-primary" : "text-muted-foreground"
              )} />
              
              <div className="space-y-2">
                <p className="text-sm font-medium">
                  Drag & drop your screenshots here
                </p>
                <p className="text-xs text-muted-foreground">
                  or click to browse files
                </p>
              </div>
              
              <input
                type="file"
                multiple
                accept={accept}
                onChange={handleFileSelect}
                className="hidden"
                id={`file-upload-${title.replace(/\s+/g, '-').toLowerCase()}`}
              />
              
              <Button 
                variant="outline" 
                asChild
                className="cursor-pointer"
              >
                <label htmlFor={`file-upload-${title.replace(/\s+/g, '-').toLowerCase()}`}>
                  <Upload className="w-4 h-4 mr-2" />
                  Choose Files
                </label>
              </Button>
              
              <p className="text-xs text-muted-foreground">
                PNG, JPG, WEBP up to 20MB each • Max {maxFiles} files
              </p>
            </div>
          </div>
          
          {files.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm font-medium">
                <CheckCircle className="w-4 h-4 text-success" />
                {files.length} file{files.length !== 1 ? 's' : ''} selected
              </div>
              
              <div className="grid grid-cols-1 gap-2 max-h-32 overflow-y-auto">
                {files.map((file, index) => (
                  <div 
                    key={index}
                    className="flex items-center gap-3 p-2 bg-muted/50 rounded-md"
                  >
                    <Image className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm flex-1 truncate">{file.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {(file.size / 1024 / 1024).toFixed(1)}MB
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(index)}
                      className="h-6 w-6 p-0 hover:bg-destructive/10 hover:text-destructive"
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};