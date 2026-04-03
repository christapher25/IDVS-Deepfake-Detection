import { useState, useRef } from "react";
import { X, Upload, CheckCircle, AlertCircle, Film, WifiOff, AlertTriangle, FileWarning, Loader2 } from "lucide-react";
import { Progress } from "../ui/progress";
import { Textarea } from "../ui/textarea";
import { Post } from "../../../App";

interface VideoUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPostCreate: (post: Post) => void;
}

type UploadStage = "idle" | "details" | "uploading" | "scanning" | "success" | "error";

export function VideoUploadModal({ isOpen, onClose, onPostCreate }: VideoUploadModalProps) {
  const [stage, setStage] = useState<UploadStage>("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [scanProgress, setScanProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [caption, setCaption] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [errorDetail, setErrorDetail] = useState("");
  const [fileType, setFileType] = useState<"image" | "video" | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- THE IRONCLAD LOCK ---
  const isCurrentlyProcessing = useRef(false);

  if (!isOpen) return null;

  const ALLOWED_VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/x-matroska"];
  const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/heic"];
  const ACCEPT_STRING = [...ALLOWED_VIDEO_TYPES, ...ALLOWED_IMAGE_TYPES].join(",");

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setErrorMessage("");
    setErrorDetail("");

    const isVideo = ALLOWED_VIDEO_TYPES.includes(file.type);
    const isImage = ALLOWED_IMAGE_TYPES.includes(file.type);

    if (!isVideo && !isImage) {
      setSelectedFile(null);
      setErrorMessage("Unsupported File");
      setErrorDetail("Please upload a valid photo (JPG, PNG) or video (MP4, MOV).");
      setStage("error");
      return;
    }

    const type = isVideo ? "video" : "image";
    setFileType(type);

    const maxSize = isVideo ? 100 * 1024 * 1024 : 15 * 1024 * 1024;
    if (file.size > maxSize) {
      setSelectedFile(null);
      setErrorMessage("File Too Large");
      setErrorDetail(`Please ensure your file is under ${isVideo ? "100MB" : "15MB"}.`);
      setStage("error");
      return;
    }

    setSelectedFile(file);
    setStage("details");
  };

  const handlePostStart = () => {
    // 1. The click lock
    if (isCurrentlyProcessing.current) return;
    isCurrentlyProcessing.current = true;

    if (!navigator.onLine) {
      setErrorMessage("No Internet Connection");
      setErrorDetail("Please check your Wi-Fi or data settings and try again.");
      setStage("error");
      isCurrentlyProcessing.current = false;
      return;
    }

    setStage("uploading");
    setUploadProgress(0);

    // 2. USE A LOCAL VARIABLE to track progress to beat the React Strict Mode bug
    let currentProgress = 0;

    const interval = setInterval(() => {
      currentProgress += 20;
      setUploadProgress(currentProgress);

      if (currentProgress >= 100) {
        clearInterval(interval);
        startScanning(); // Safely outside the React state updater
      }
    }, 200);
  };

  const startScanning = async () => {
    setStage("scanning");
    setScanProgress(0);

    const formData = new FormData();
    if(selectedFile) formData.append("file", selectedFile);

    const controller = new AbortController();
    // CHANGED: Increased from 30,000 (30s) to 300,000 (5 minutes)
    const timeoutId = setTimeout(() => controller.abort(), 300000);

    try {
      const progressInt = setInterval(() => {
        setScanProgress((prev) => (prev >= 90 ? 90 : prev + 5));
      }, 400);

      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      clearInterval(progressInt);
      setScanProgress(100);

      if (!response.ok) {
        throw new Error(`Server Error (${response.status})`);
      }

      const data = await response.json();

      const newPost: Post = {
        id: Date.now(),
        username: "current_user",
        userAvatar: "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100",
        location: "Kerala, India",
        mediaUrl: URL.createObjectURL(selectedFile!),
        mediaType: fileType || "image",
        likes: 0,
        caption: caption,
        comments: 0,
        timeAgo: "JUST NOW",
        isDeepfake: data.result === "FAKE",
        deepfakeReason: data.reason
      };

      onPostCreate(newPost);
      setStage("success");
      isCurrentlyProcessing.current = false; // Unlock on success

    } catch (error: any) {
      isCurrentlyProcessing.current = false; // Unlock on error
      setStage("error");
      if (error.name === 'AbortError') {
        setErrorMessage("Request Timeout");
        setErrorDetail("The analysis is taking longer than expected. Please try again.");
      } else if (error.message.includes("Failed to fetch")) {
        setErrorMessage("AI Service Unavailable");
        setErrorDetail("We couldn't connect to the analysis server. Please try again in a moment.");
      } else {
        setErrorMessage("Upload Failed");
        setErrorDetail("Something went wrong while processing your file.");
      }
    }
  };

  const handleReset = () => {
    setStage("idle");
    setUploadProgress(0);
    setScanProgress(0);
    setSelectedFile(null);
    setCaption("");
    setErrorMessage("");
    if (fileInputRef.current) fileInputRef.current.value = "";
    isCurrentlyProcessing.current = false; // Unlock on reset
  };

  const handleFullClose = () => {
    handleReset();
    onClose();
  };

  const isFileError = errorMessage === "Unsupported File" || errorMessage === "File Too Large";
  const isProcessing = stage === "uploading" || stage === "scanning";

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
      <div className="bg-white rounded-xl w-full max-w-lg relative overflow-hidden flex flex-col max-h-[90vh]">

        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold">Create new post</h2>
          <button onClick={handleFullClose} className="p-1 hover:bg-gray-100 rounded-full transition">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto">
          {/* STEP 1: SELECT */}
          {stage === "idle" && (
            <div className="flex flex-col items-center gap-6 py-8">
              <div className="w-20 h-20 rounded-full bg-slate-100 flex items-center justify-center">
                <Film className="w-10 h-10 text-slate-400" />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-medium mb-1">Drag photos and videos here</h3>
                <p className="text-gray-400 text-xs">Supported: MP4, MOV, JPG, PNG</p>
              </div>
              <input ref={fileInputRef} type="file" accept={ACCEPT_STRING} onChange={handleFileSelect} className="hidden" />
              <button onClick={() => fileInputRef.current?.click()} className="bg-[#0095f6] text-white px-4 py-2 rounded-md text-sm font-semibold hover:bg-[#1877f2] transition">
                Select from computer
              </button>
            </div>
          )}

          {/* STEP 2: DETAILS */}
          {stage === "details" && selectedFile && (
            <div className="flex flex-col gap-4">
              <div className="flex gap-4 items-start">
                <div className="w-16 h-16 bg-gray-100 rounded-md overflow-hidden flex-shrink-0">
                  {fileType === "video" ? (
                    <video src={URL.createObjectURL(selectedFile)} className="w-full h-full object-cover" />
                  ) : (
                    <img src={URL.createObjectURL(selectedFile)} className="w-full h-full object-cover" />
                  )}
                </div>
                <div className="flex-1 overflow-hidden">
                  <p className="text-sm font-medium truncate">{selectedFile.name}</p>
                  <p className="text-xs text-gray-500">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Caption</label>
                <Textarea
                  placeholder="Write a caption..."
                  value={caption}
                  onChange={(e) => setCaption(e.target.value)}
                  className="min-h-[100px]"
                />
              </div>

              <button
                onClick={handlePostStart}
                disabled={isProcessing}
                className={`w-full text-white py-2.5 rounded-md text-sm font-semibold transition mt-2 flex items-center justify-center gap-2
                  ${isProcessing ? "bg-blue-300 cursor-not-allowed" : "bg-[#0095f6] hover:bg-[#1877f2]"}`}
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Share"
                )}
              </button>
            </div>
          )}

          {/* STEP 3: SCAN */}
          {(stage === "uploading" || stage === "scanning") && (
            <div className="flex flex-col items-center gap-6 py-8">
              <div className="w-full space-y-4">
                <div className="flex justify-between text-xs font-medium uppercase tracking-wide text-gray-500">
                  <span>{stage === "uploading" ? "Uploading Media" : "AI Security Scan"}</span>
                  <span>{stage === "uploading" ? uploadProgress : scanProgress}%</span>
                </div>
                <Progress value={stage === "uploading" ? uploadProgress : scanProgress} className={`h-2 ${stage === "scanning" ? "bg-purple-100" : "bg-blue-100"}`} />
                <p className="text-center text-xs text-gray-400">
                  {stage === "uploading" ? "Please wait while we upload your file..." : "Checking for Deepfake signatures & liveness..."}
                </p>
              </div>
            </div>
          )}

          {/* STEP 4: SUCCESS */}
          {stage === "success" && (
            <div className="flex flex-col items-center gap-4 py-6">
              <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center animate-in zoom-in">
                <CheckCircle className="w-8 h-8 text-green-600" />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900">Shared Successfully</h3>
                <p className="text-sm text-gray-500">Your post is now live on the feed.</p>
              </div>
              <button onClick={handleFullClose} className="mt-4 text-blue-500 font-semibold text-sm hover:text-blue-700">Close</button>
            </div>
          )}

          {/* STEP 5: SMART ERROR */}
          {stage === "error" && (
            <div className="flex flex-col items-center gap-4 py-6">
              <div className="w-16 h-16 rounded-full bg-red-50 flex items-center justify-center animate-in shake">
                {isFileError ? (
                   <FileWarning className="w-8 h-8 text-red-500" />
                ) : errorMessage.includes("Internet") ? (
                   <WifiOff className="w-8 h-8 text-red-500" />
                ) : (
                   <AlertTriangle className="w-8 h-8 text-red-500" />
                )}
              </div>
              <div className="text-center px-4">
                <h3 className="text-lg font-semibold text-red-600">{errorMessage}</h3>
                <p className="text-sm text-gray-600 mt-2 bg-gray-50 p-3 rounded-md border border-gray-100">{errorDetail}</p>
              </div>

              <div className="flex gap-3 w-full mt-2">
                {isFileError ? (
                  <button onClick={handleReset} className="w-full bg-[#0095f6] text-white py-2.5 rounded-md text-sm font-semibold hover:bg-[#1877f2]">
                    Select Different File
                  </button>
                ) : (
                  <>
                    <button onClick={() => setStage("details")} className="flex-1 bg-gray-100 text-gray-700 py-2.5 rounded-md text-sm font-semibold hover:bg-gray-200">Back</button>
                    <button onClick={handlePostStart} className="flex-1 bg-blue-500 text-white py-2.5 rounded-md text-sm font-semibold hover:bg-blue-600">Retry</button>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}