import React, { useEffect, useRef, useState, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { Mic, MicOff, Video, VideoOff, PhoneOff, Loader2, AlertCircle, Play, Briefcase, User, Layers, AlertTriangle, Wand2, ScrollText, Eye, Square, Pause, Circle, Download, Trash2, RefreshCw, X, ScanFace, Smartphone, Monitor } from 'lucide-react';
import { ConnectionState, AppMode } from '../types';
import { createPcmBlob, decode, decodeAudioData, blobToBase64 } from '../utils/audio';
import AudioVisualizer from './AudioVisualizer';

const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';
const SCRIPT_MODEL_NAME = 'gemini-2.5-flash';
const FRAME_RATE = 5; // FPS for video streaming
const JPEG_QUALITY = 0.8;

const LiveSession: React.FC = () => {
  const [isLandscape, setIsLandscape] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [mode, setMode] = useState<AppMode>(AppMode.CONVERSATION);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  
  // Controls
  const [isMicOn, setIsMicOn] = useState(true);
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [isAiMuted, setIsAiMuted] = useState(false);
  const [isGazeCorrectionActive, setIsGazeCorrectionActive] = useState(true);

  // Teleprompter State
  const [scriptText, setScriptText] = useState('');
  const [scriptTopic, setScriptTopic] = useState('');
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [scrollSpeed, setScrollSpeed] = useState(1); // 0 to 5
  const [fontSize, setFontSize] = useState(typeof window !== 'undefined' && window.innerWidth < 768 ? 28 : 42);
  const [textPosition, setTextPosition] = useState(40); // vh from top
  const [voiceScrollEnabled, setVoiceScrollEnabled] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const timerRef = useRef<number | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Recording State
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const [recordedVideoUrl, setRecordedVideoUrl] = useState<string | null>(null);
  const [recordingMimeType, setRecordingMimeType] = useState<string>('');
  const [isReviewing, setIsReviewing] = useState(false);

  // State
  const [currentInputText, setCurrentInputText] = useState('');
  const [currentOutputText, setCurrentOutputText] = useState('');

  // Refs for audio/video processing
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<any>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const inputContextRef = useRef<AudioContext | null>(null);
  const outputGainRef = useRef<GainNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  
  // Media Streams
  const micStreamRef = useRef<MediaStream | null>(null);
  const sysStreamRef = useRef<MediaStream | null>(null);
  
  // Processing Nodes
  const videoIntervalRef = useRef<number | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const sysSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const scrollIntervalRef = useRef<number | null>(null);
  
  // Analysers for visualization
  const [inputAnalyser, setInputAnalyser] = useState<AnalyserNode | null>(null);
  const [outputAnalyser, setOutputAnalyser] = useState<AnalyserNode | null>(null);

  const stopAudio = useCallback(() => {
    sourcesRef.current.forEach(source => {
      try { source.stop(); } catch (e) { /* ignore */ }
    });
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;
  }, []);

  const cleanup = useCallback((preserveReview: boolean = false) => {
    if (videoIntervalRef.current) clearInterval(videoIntervalRef.current);
    if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current);
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setRecordingSeconds(0);
    
    if (processorRef.current) {
      try { processorRef.current.disconnect(); } catch (e) { /* ignore */ }
      processorRef.current = null;
    }
    
    // Disconnect sources
    if (micSourceRef.current) { try { micSourceRef.current.disconnect(); } catch(e){} micSourceRef.current = null; }
    if (sysSourceRef.current) { try { sysSourceRef.current.disconnect(); } catch(e){} sysSourceRef.current = null; }
    
    // Stop streams
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach(track => track.stop());
      micStreamRef.current = null;
    }
    if (sysStreamRef.current) {
      sysStreamRef.current.getTracks().forEach(track => track.stop());
      sysStreamRef.current = null;
    }

    if (inputContextRef.current) {
      try { inputContextRef.current.close(); } catch (e) { /* ignore */ }
      inputContextRef.current = null;
    }
    
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (e) { /* ignore */ }
      audioContextRef.current = null;
    }

    if (sessionRef.current) {
      sessionRef.current = null;
    }

    // Stop recorder if still active (fallback)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }

    setConnectionState(ConnectionState.DISCONNECTED);
    setInputAnalyser(null);
    setOutputAnalyser(null);
    setCurrentInputText('');
    setCurrentOutputText('');
    setWarning(null);
    setIsRecording(false);
    setIsPaused(false);
    
    if (!preserveReview) {
      setIsReviewing(false);
      setRecordedVideoUrl(null);
      setRecordingMimeType('');
    }
  }, []);

  const formatDuration = (secs: number) => {
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  };

  const startRecording = () => {
    if (!micStreamRef.current) {
      setWarning("No camera/microphone stream available.");
      return;
    }
    
    try {
      const stream = micStreamRef.current;
      
      const mimeTypes = [
        'video/mp4',
        'video/webm;codecs=h264',
        'video/webm;codecs=vp9',
        'video/webm'
      ];
      
      let selectedType = '';
      for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          selectedType = type;
          break;
        }
      }
      
      const options = selectedType ? { mimeType: selectedType } : undefined;
      setRecordingMimeType(selectedType || 'video/webm');

      const recorder = new MediaRecorder(stream, options);
      
      recordedChunksRef.current = [];
      
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };
      
      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: selectedType || 'video/webm' });
        if (blob.size > 0) {
          const url = URL.createObjectURL(blob);
          setRecordedVideoUrl(url);
          setIsReviewing(true);
          cleanup(true);
        } else {
          cleanup(false);
        }
      };
      
      recorder.start(1000); // 1s chunks
      mediaRecorderRef.current = recorder;
      setRecordingSeconds(0);
      timerRef.current = window.setInterval(() => setRecordingSeconds(s => s + 1), 1000);
      setIsRecording(true);
      setIsPaused(false);
    } catch (e) {
      console.error("Recording failed to start", e);
      // Fallback logic
      try {
         const stream = micStreamRef.current;
         const recorder = new MediaRecorder(stream);
         setRecordingMimeType('video/webm');
         recordedChunksRef.current = [];
         recorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunksRef.current.push(e.data); };
         recorder.onstop = () => {
            const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
            if (blob.size > 0) {
              const url = URL.createObjectURL(blob);
              setRecordedVideoUrl(url);
              setIsReviewing(true);
              cleanup(true);
            } else {
              cleanup(false);
            }
         };
         recorder.start(1000);
         mediaRecorderRef.current = recorder;
         setRecordingSeconds(0);
         timerRef.current = window.setInterval(() => setRecordingSeconds(s => s + 1), 1000);
         setIsRecording(true);
         setIsPaused(false);
      } catch (retryError) {
         setWarning("Recording not supported on this device.");
      }
    }
  };

  const stopRecording = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    } else {
      if (recordedChunksRef.current.length > 0) {
         const blob = new Blob(recordedChunksRef.current, { type: recordingMimeType || 'video/webm' });
         const url = URL.createObjectURL(blob);
         setRecordedVideoUrl(url);
         setIsReviewing(true);
         cleanup(true);
      } else {
         cleanup(false);
      }
    }
  };

  const togglePauseRecording = () => {
    if (!mediaRecorderRef.current) return;
    if (isPaused) {
      mediaRecorderRef.current.resume();
      timerRef.current = window.setInterval(() => setRecordingSeconds(s => s + 1), 1000);
      setIsPaused(false);
    } else {
      mediaRecorderRef.current.pause();
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      setIsPaused(true);
    }
  };

  const discardRecording = () => {
    if (recordedVideoUrl) {
      URL.revokeObjectURL(recordedVideoUrl);
      setRecordedVideoUrl(null);
    }
    setIsReviewing(false);
    cleanup(false);
  };

  const generateScript = async () => {
    if (!scriptTopic.trim()) return;
    setIsGeneratingScript(true);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const response = await ai.models.generateContent({
        model: SCRIPT_MODEL_NAME,
        contents: `Write a professional, engaging script for a video recording about: "${scriptTopic}". 
        Keep it under 200 words. Write it in the first person. Do not include scene directions, just the spoken text.
        Format it with line breaks for easy reading.`,
      });
      setScriptText(response.text || "");
    } catch (e) {
      console.error(e);
      setWarning("Failed to generate script. Please try again.");
    } finally {
      setIsGeneratingScript(false);
    }
  };

  const startSession = async () => {
    try {
      setConnectionState(ConnectionState.CONNECTING);
      setError(null);
      setWarning(null);
      setRecordedVideoUrl(null);
      setIsReviewing(false);

      // Initialize Audio Contexts
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioContextClass({ sampleRate: 24000 });
      const inputCtx = new AudioContextClass({ sampleRate: 16000 });
      
      audioContextRef.current = audioCtx;
      inputContextRef.current = inputCtx;

      // Create Gain Node for output volume control
      const gainNode = audioCtx.createGain();
      gainNode.gain.value = isAiMuted ? 0 : 1;
      gainNode.connect(audioCtx.destination);
      outputGainRef.current = gainNode;

      // Output Visualizer
      const outAnalyser = audioCtx.createAnalyser();
      outAnalyser.fftSize = 256;
      outAnalyser.connect(gainNode);
      setOutputAnalyser(outAnalyser);

      // 1. Get Microphone Stream
      let micStream: MediaStream;
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
          video: { width: { ideal: 1280 }, height: { ideal: 720 } }
        });
      } catch (err: any) {
        throw new Error("Permission denied. Please allow camera and microphone access.");
      }
      micStreamRef.current = micStream;

      // Video Setup (User Face)
      if (videoRef.current) {
        videoRef.current.srcObject = micStream;
        videoRef.current.play().catch(e => console.warn("Video play failed", e));
      }

      // Input Visualizer & Processing
      const inAnalyser = inputCtx.createAnalyser();
      inAnalyser.fftSize = 256;
      setInputAnalyser(inAnalyser);

      const micSource = inputCtx.createMediaStreamSource(micStream);
      micSourceRef.current = micSource;
      micSource.connect(inAnalyser);

      // --- Teleprompter Mode Logic ---
      if (mode === AppMode.TELEPROMPTER) {
        setConnectionState(ConnectionState.CONNECTED);
        setIsRecording(false);
        setIsPaused(false);
        return; 
      }

      // --- Copilot System Audio ---
      let sysStream: MediaStream | null = null;
      if (mode === AppMode.COPILOT) {
        try {
          sysStream = await navigator.mediaDevices.getDisplayMedia({
            video: true, 
            audio: true
          });
          if (sysStream.getAudioTracks().length > 0) {
             sysStreamRef.current = sysStream;
             const sysSource = inputCtx.createMediaStreamSource(sysStream);
             sysSourceRef.current = sysSource;
             // Mix System Audio into input analyser and processor
             sysSource.connect(inAnalyser);
          } else {
             setWarning("System audio not shared. The AI might struggle to hear the interviewer.");
             sysStream.getTracks().forEach(t => t.stop());
             sysStream = null;
          }
        } catch (err: any) {
          console.warn("System audio capture failed", err);
          setWarning("Screen sharing cancelled. Using microphone only.");
        }
      }

      // --- Gemini Live Connection (Conversation & Copilot only) ---
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      let systemInstructionText = "You are Clueku, a helpful AI assistant.";
      if (mode === AppMode.COPILOT) {
        systemInstructionText = `
          You are an expert interview copilot. 
          1. Listen for questions.
          2. Provide concise, professional answers in the first person.
          3. Keep responses short and punchy.
        `;
      }

      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: { parts: [{ text: systemInstructionText }] },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setConnectionState(ConnectionState.CONNECTED);
            const processor = inputCtx.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;
            micSource.connect(processor);
            if (sysSourceRef.current) sysSourceRef.current.connect(processor);

            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createPcmBlob(inputData);
              sessionPromise.then(session => {
                 session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            processor.connect(inputCtx.destination);

            videoIntervalRef.current = window.setInterval(() => {
               if (!isCameraOn || !videoRef.current || !canvasRef.current) return;
               const canvas = canvasRef.current;
               const video = videoRef.current;
               const ctx = canvas.getContext('2d');
               if (video.readyState === video.HAVE_ENOUGH_DATA && ctx) {
                 canvas.width = video.videoWidth;
                 canvas.height = video.videoHeight;
                 ctx.drawImage(video, 0, 0);
                 canvas.toBlob(async (blob) => {
                   if (blob) {
                     const base64 = await blobToBase64(blob);
                     sessionPromise.then(session => {
                       session.sendRealtimeInput({
                         media: { mimeType: 'image/jpeg', data: base64 }
                       });
                     });
                   }
                 }, 'image/jpeg', JPEG_QUALITY);
               }
            }, 1000 / FRAME_RATE);
          },
          onmessage: async (msg: LiveServerMessage) => {
            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData) {
              const ctx = audioContextRef.current;
              if (ctx && outAnalyser) {
                 nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
                 const audioBuffer = await decodeAudioData(decode(audioData), ctx, 24000, 1);
                 const source = ctx.createBufferSource();
                 source.buffer = audioBuffer;
                 source.connect(outAnalyser);
                 source.onended = () => sourcesRef.current.delete(source);
                 source.start(nextStartTimeRef.current);
                 nextStartTimeRef.current += audioBuffer.duration;
                 sourcesRef.current.add(source);
              }
            }
            if (msg.serverContent?.outputTranscription?.text) {
               setCurrentOutputText(prev => prev + msg.serverContent!.outputTranscription!.text);
            }
            if (msg.serverContent?.inputTranscription?.text) {
               setCurrentInputText(prev => prev + msg.serverContent!.inputTranscription!.text);
            }
            if (msg.serverContent?.interrupted) {
              stopAudio();
              setCurrentOutputText(''); 
            }
          },
          onclose: () => setConnectionState(ConnectionState.DISCONNECTED),
          onerror: (e) => {
            console.error(e);
            setConnectionState(ConnectionState.ERROR);
          }
        }
      });
      sessionRef.current = sessionPromise;

    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to start session");
      setConnectionState(ConnectionState.ERROR);
      cleanup(false);
    }
  };

  // Smart Scroll Logic
  useEffect(() => {
    if (mode !== AppMode.TELEPROMPTER || connectionState !== ConnectionState.CONNECTED || !isRecording || isPaused) {
      if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current);
      return;
    }
    const checkAudioAndScroll = () => {
      if (!scrollContainerRef.current) return;
      if (!voiceScrollEnabled) {
        scrollContainerRef.current.scrollTop += scrollSpeed;
        return;
      }
      if (!inputAnalyser) return;
      const bufferLength = inputAnalyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      inputAnalyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for(let i = 0; i < bufferLength; i++) { sum += dataArray[i]; }
      const average = sum / bufferLength;
      if (average > 25) {
        scrollContainerRef.current.scrollTop += scrollSpeed;
      }
    };
    scrollIntervalRef.current = window.setInterval(checkAudioAndScroll, 30);
    return () => {
      if (scrollIntervalRef.current) clearInterval(scrollIntervalRef.current);
    };
  }, [mode, connectionState, isRecording, isPaused, inputAnalyser, scrollSpeed, voiceScrollEnabled]);


  useEffect(() => {
    if (outputGainRef.current) {
      outputGainRef.current.gain.setValueAtTime(isAiMuted ? 0 : 1, audioContextRef.current?.currentTime || 0);
    }
  }, [isAiMuted]);

  const toggleMic = () => {
    setIsMicOn(!isMicOn);
    if (micStreamRef.current) micStreamRef.current.getAudioTracks().forEach(t => t.enabled = !isMicOn);
  };

  const toggleCamera = () => {
    setIsCameraOn(!isCameraOn);
    if (micStreamRef.current) micStreamRef.current.getVideoTracks().forEach(t => t.enabled = !isCameraOn);
  };

  // --- Render Logic ---

  // 1. Review Screen
  if (isReviewing && recordedVideoUrl) {
    return (
       <div className="relative w-full h-screen bg-slate-950 flex flex-col items-center justify-center p-4 sm:p-6 animate-fade-in">
          <div className="w-full max-w-4xl bg-slate-900 rounded-3xl border border-slate-800 p-4 sm:p-8 shadow-2xl flex flex-col items-center">
            <h2 className="text-2xl sm:text-3xl font-bold text-white mb-4 sm:mb-6 flex items-center gap-3">
              <Video className="w-8 h-8 text-emerald-500" />
              Recording Preview
            </h2>
            
            <div className="w-full aspect-video bg-black rounded-2xl overflow-hidden border border-slate-700 mb-8 shadow-lg">
              <video 
                src={recordedVideoUrl} 
                controls 
                className="w-full h-full object-contain" 
              />
            </div>

            <div className="flex flex-col md:flex-row gap-4 w-full justify-center">
              <button 
                onClick={discardRecording}
                className="px-6 py-4 rounded-xl bg-slate-800 hover:bg-red-900/30 text-slate-300 hover:text-red-400 font-medium transition-all flex items-center justify-center gap-2 group border border-transparent hover:border-red-500/30"
              >
                <Trash2 className="w-5 h-5 group-hover:scale-110 transition-transform" />
                Discard & Try Again
              </button>
              
              {/* Force .mp4 extension regardless of container */}
              <a 
                href={recordedVideoUrl} 
                download={`clueku-recording-${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.mp4`}
                className="px-8 py-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-lg transition-all shadow-lg shadow-emerald-900/20 flex items-center justify-center gap-3 hover:scale-105"
              >
                <Download className="w-6 h-6" />
                Save Recording (.mp4)
              </a>
            </div>
          </div>
       </div>
    );
  }

  const renderStartScreen = () => (
    <div className={`flex flex-col items-center w-full max-w-5xl animate-fade-in px-4 pb-safe ${isLandscape ? 'overflow-y-auto max-h-screen py-2' : ''}`}>
      <h1 className={`font-bold text-white text-center leading-tight ${isLandscape ? 'text-2xl mb-2' : 'text-3xl md:text-6xl mb-4'}`}>
        Choose Your Interface
      </h1>
      <div className={`grid gap-3 w-full ${isLandscape ? 'grid-cols-3 mt-2' : 'grid-cols-1 sm:grid-cols-3 mt-6'}`}>
        <button
          onClick={() => setMode(AppMode.CONVERSATION)}
          className={`group relative rounded-3xl border transition-all duration-300 text-left active:scale-95 hover:scale-[1.02] flex items-center gap-3 ${isLandscape ? 'p-3 flex-col justify-center' : 'p-4 sm:p-6 sm:flex-col sm:items-start'} ${mode === AppMode.CONVERSATION ? 'bg-blue-600/20 border-blue-500 ring-2 ring-blue-500/50' : 'bg-white/5 border-white/10 hover:bg-white/10'}`}
        >
          <div className={`shrink-0 bg-blue-500 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/30 ${isLandscape ? 'w-9 h-9' : 'w-12 h-12'}`}>
            <User className={isLandscape ? 'w-5 h-5 text-white' : 'w-6 h-6 text-white'} />
          </div>
          <div>
            <h3 className={`font-bold ${isLandscape ? 'text-sm text-center' : 'text-base sm:text-xl sm:mb-2'}`}>Conversation</h3>
            {!isLandscape && <p className="text-slate-400 text-sm hidden sm:block">Natural video chat with Gemini.</p>}
          </div>
        </button>

        <button
          onClick={() => setMode(AppMode.COPILOT)}
          className={`group relative rounded-3xl border transition-all duration-300 text-left active:scale-95 hover:scale-[1.02] flex items-center gap-3 ${isLandscape ? 'p-3 flex-col justify-center' : 'p-4 sm:p-6 sm:flex-col sm:items-start'} ${mode === AppMode.COPILOT ? 'bg-purple-600/20 border-purple-500 ring-2 ring-purple-500/50' : 'bg-white/5 border-white/10 hover:bg-white/10'}`}
        >
          <div className={`shrink-0 bg-purple-500 rounded-2xl flex items-center justify-center shadow-lg shadow-purple-500/30 ${isLandscape ? 'w-9 h-9' : 'w-12 h-12'}`}>
            <Briefcase className={isLandscape ? 'w-5 h-5 text-white' : 'w-6 h-6 text-white'} />
          </div>
          <div>
            <h3 className={`font-bold ${isLandscape ? 'text-sm text-center' : 'text-base sm:text-xl sm:mb-2'}`}>Interview Copilot</h3>
            {!isLandscape && <p className="text-slate-400 text-sm hidden sm:block">Real-time interview assistance.</p>}
          </div>
          <div className="absolute top-2 right-2 bg-purple-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded uppercase">Pro</div>
        </button>

        <button
          onClick={() => setMode(AppMode.TELEPROMPTER)}
          className={`group relative rounded-3xl border transition-all duration-300 text-left active:scale-95 hover:scale-[1.02] flex items-center gap-3 ${isLandscape ? 'p-3 flex-col justify-center' : 'p-4 sm:p-6 sm:flex-col sm:items-start'} ${mode === AppMode.TELEPROMPTER ? 'bg-emerald-600/20 border-emerald-500 ring-2 ring-emerald-500/50' : 'bg-white/5 border-white/10 hover:bg-white/10'}`}
        >
          <div className={`shrink-0 bg-emerald-500 rounded-2xl flex items-center justify-center shadow-lg shadow-emerald-500/30 ${isLandscape ? 'w-9 h-9' : 'w-12 h-12'}`}>
            <ScrollText className={isLandscape ? 'w-5 h-5 text-white' : 'w-6 h-6 text-white'} />
          </div>
          <div>
            <h3 className={`font-bold ${isLandscape ? 'text-sm text-center' : 'text-base sm:text-xl sm:mb-2'}`}>Teleprompter</h3>
            {!isLandscape && <p className="text-slate-400 text-sm hidden sm:block">AI Scripting & Smart Scroll.</p>}
          </div>
        </button>
      </div>

      {mode === AppMode.TELEPROMPTER && (
        <div className={`w-full max-w-3xl bg-slate-900/50 border border-emerald-500/30 rounded-2xl animate-in slide-in-from-bottom-4 ${isLandscape ? 'mt-2 p-3' : 'mt-6 p-4 sm:p-6'}`}>
           <div className={`flex gap-3 ${isLandscape ? 'flex-row items-start' : 'flex-col'}`}>
             <div className={`flex flex-col gap-2 ${isLandscape ? 'flex-1' : 'w-full'}`}>
               <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
                 <Wand2 className="w-4 h-4 text-emerald-400" />
                 Generate Script with AI
               </label>
               <div className="flex gap-2">
                 <input
                   type="text"
                   value={scriptTopic}
                   onChange={(e) => setScriptTopic(e.target.value)}
                   placeholder="Topic..."
                   className="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 focus:ring-2 focus:ring-emerald-500 outline-none transition-all text-sm"
                   onKeyDown={(e) => e.key === 'Enter' && generateScript()}
                 />
                 <button
                   onClick={generateScript}
                   disabled={isGeneratingScript || !scriptTopic}
                   className="bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white px-4 py-2 rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm"
                 >
                   {isGeneratingScript ? <Loader2 className="w-4 h-4 animate-spin"/> : "Generate"}
                 </button>
               </div>
               <textarea
                 value={scriptText}
                 onChange={(e) => setScriptText(e.target.value)}
                 placeholder="Or paste your script here..."
                 className={`w-full bg-slate-950 border border-slate-700 rounded-xl p-3 focus:ring-2 focus:ring-emerald-500 outline-none transition-all font-mono text-sm leading-relaxed ${isLandscape ? 'h-24' : 'h-36 sm:h-48'}`}
               />
             </div>
             <div className={`flex ${isLandscape ? 'flex-col justify-end' : 'justify-end pt-2'}`}>
               <button
                 onClick={startSession}
                 disabled={!scriptText.trim()}
                 className="bg-emerald-500 hover:bg-emerald-400 text-white px-6 py-3 rounded-xl font-bold shadow-lg shadow-emerald-500/20 flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 text-sm whitespace-nowrap"
               >
                 <Video className="w-4 h-4" />
                 Start
               </button>
             </div>
           </div>
        </div>
      )}

      {mode !== AppMode.TELEPROMPTER && (
        <button
          onClick={startSession}
          className="mt-8 sm:mt-12 px-10 py-4 bg-white text-slate-950 rounded-full font-bold text-lg hover:bg-blue-50 active:scale-95 transition-all hover:scale-105 shadow-xl hover:shadow-blue-500/25 flex items-center gap-3 min-h-[56px]"
        >
            <Play className="w-5 h-5 fill-current"/>
            Start Session
        </button>
      )}
    </div>
  );

  return (
    <div className="relative w-full h-screen bg-slate-950 flex flex-col overflow-hidden font-sans text-slate-100">
      <div className="absolute inset-0 z-0">
         <div className="absolute inset-0 bg-gradient-to-b from-slate-900/90 via-slate-900/80 to-slate-950 pointer-events-none" />
         <video 
           ref={videoRef} 
           className={`w-full h-full object-cover transition-opacity duration-500 ${connectionState === ConnectionState.CONNECTED ? 'opacity-100' : 'opacity-20'}`}
           autoPlay 
           playsInline 
           muted
           style={{ transform: 'scaleX(-1)' }} 
         />
         
         {/* AI Gaze Correction Simulation Overlay (Mocking the effect) */}
         {mode === AppMode.TELEPROMPTER && connectionState === ConnectionState.CONNECTED && isGazeCorrectionActive && (
            <div className="absolute inset-0 pointer-events-none z-10 flex items-center justify-center">
               {/* Subtle face tracking reticle to imply AI processing */}
               <div className="w-64 h-64 border border-emerald-500/20 rounded-full absolute animate-pulse opacity-20" />
               <div className="w-60 h-60 border-t border-b border-emerald-500/30 rounded-full absolute animate-spin-slow opacity-20" />
            </div>
         )}

         <canvas ref={canvasRef} className="hidden" />
      </div>

      <div className="absolute top-0 left-0 right-0 px-4 py-4 sm:p-6 z-20 flex justify-between items-center pointer-events-none">
        <div className="flex items-center gap-2 pointer-events-auto">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-xl tracking-tight">Clueku</span>
        </div>
        <button
          onClick={() => setIsLandscape(l => !l)}
          className="pointer-events-auto flex items-center gap-1.5 bg-white/10 hover:bg-white/20 active:scale-95 border border-white/20 px-3 py-1.5 rounded-full text-xs font-medium text-white transition-all"
          title="Toggle layout orientation"
        >
          {isLandscape ? <Monitor className="w-3.5 h-3.5" /> : <Smartphone className="w-3.5 h-3.5" />}
          {isLandscape ? 'Landscape' : 'Portrait'}
        </button>
        
        <div className="flex items-center gap-4 pointer-events-auto">
          {warning && (
            <div className="flex bg-yellow-500/10 border border-yellow-500/30 px-3 py-1.5 rounded-full text-xs font-medium text-yellow-300 items-center gap-2 animate-in fade-in slide-in-from-top-2 max-w-[180px] sm:max-w-none truncate">
              <AlertTriangle className="w-3 h-3" />
              {warning}
            </div>
          )}
          {mode === AppMode.TELEPROMPTER && connectionState === ConnectionState.CONNECTED && (
             <>
                <button 
                   onClick={() => setIsGazeCorrectionActive(!isGazeCorrectionActive)}
                   className={`px-4 py-1.5 rounded-full text-xs font-bold flex items-center gap-2 border transition-all ${
                     isGazeCorrectionActive 
                       ? 'bg-emerald-500/20 border-emerald-500 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.2)]' 
                       : 'bg-slate-800/50 border-slate-700 text-slate-400'
                   }`}
                >
                  <ScanFace className={`w-3 h-3 ${isGazeCorrectionActive ? 'animate-pulse' : ''}`} />
                  {isGazeCorrectionActive ? 'AI Gaze: ACTIVE' : 'AI Gaze: OFF'}
                </button>
                
                <button 
                  onClick={() => cleanup(false)}
                  className="bg-red-500/20 hover:bg-red-500/40 text-red-200 p-2 rounded-full transition-all border border-red-500/30"
                  title="Cancel & Exit"
                >
                  <X className="w-5 h-5" />
                </button>
             </>
          )}
        </div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col justify-center items-center w-full mx-auto">
        
        {connectionState === ConnectionState.DISCONNECTED && renderStartScreen()}

        {connectionState === ConnectionState.CONNECTING && (
          <div className="flex flex-col items-center gap-6">
            <div className="relative">
              <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="w-6 h-6 text-blue-500 animate-pulse" />
              </div>
            </div>
            <p className="text-slate-300 font-mono text-sm tracking-widest uppercase">Initializing...</p>
          </div>
        )}

        {connectionState === ConnectionState.CONNECTED && (
          <div className="w-full h-full flex flex-col">
            
            {mode === AppMode.TELEPROMPTER && (
              <>
                {/* SCRIPT OVERLAY - Relaxed mask to reveal more text */}
                <div className="absolute inset-x-0 top-0 h-full flex justify-center z-30 pointer-events-none">
                   <div
                      ref={scrollContainerRef}
                      className="w-full max-w-5xl overflow-hidden text-center relative pointer-events-auto"
                      style={{
                        paddingTop: `${textPosition}vh`,
                        maskImage: 'linear-gradient(to bottom, black 0%, black 80%, transparent 100%)',
                        WebkitMaskImage: 'linear-gradient(to bottom, black 0%, black 80%, transparent 100%)'
                      }}
                   >
                     <p 
                       className="font-bold text-white drop-shadow-2xl leading-relaxed transition-all duration-300 whitespace-pre-wrap px-12"
                       style={{ fontSize: `${fontSize}px`, textShadow: '0 4px 8px rgba(0,0,0,0.9)' }}
                     >
                       {scriptText}
                     </p>
                     <div className="h-[120vh]" /> 
                   </div>
                </div>

                {/* Landscape: right-side panel | Portrait: bottom bar */}
                <div className={`absolute z-50 flex gap-3 ${
                  isLandscape
                    ? 'right-3 top-1/2 -translate-y-1/2 flex-col items-center'
                    : 'bottom-4 left-1/2 -translate-x-1/2 flex-col items-center w-full max-w-lg px-4'
                }`}>

                  {/* Sliders */}
                  <div className={`flex bg-black/60 backdrop-blur-md rounded-2xl border border-white/10 ${
                    isLandscape ? 'flex-col gap-3 px-3 py-4 items-start' : 'flex-wrap items-center justify-center gap-4 px-4 py-3 w-full'
                  }`}>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-400 font-mono uppercase">Speed</span>
                      <input
                        type="range" min="0.1" max="5" step="0.1"
                        value={scrollSpeed}
                        onChange={(e) => setScrollSpeed(parseFloat(e.target.value))}
                        className={`accent-emerald-500 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer ${isLandscape ? 'w-16' : 'w-20 sm:w-24'}`}
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-400 font-mono uppercase">Size</span>
                      <input
                        type="range" min="12" max="72" step="2"
                        value={fontSize}
                        onChange={(e) => setFontSize(parseInt(e.target.value))}
                        className={`accent-emerald-500 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer ${isLandscape ? 'w-16' : 'w-20 sm:w-24'}`}
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-400 font-mono uppercase">Height</span>
                      <input
                        type="range" min="5" max="75" step="5"
                        value={textPosition}
                        onChange={(e) => setTextPosition(parseInt(e.target.value))}
                        className={`accent-emerald-500 h-2 bg-white/20 rounded-lg appearance-none cursor-pointer ${isLandscape ? 'w-16' : 'w-20 sm:w-24'}`}
                      />
                    </div>
                    {/* Voice scroll toggle */}
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-400 font-mono uppercase">Voice</span>
                      <button
                        onClick={() => setVoiceScrollEnabled(v => !v)}
                        className={`relative w-8 h-4 rounded-full transition-colors duration-300 ${voiceScrollEnabled ? 'bg-emerald-500' : 'bg-slate-600'}`}
                      >
                        <span className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-white shadow transition-transform duration-300 ${voiceScrollEnabled ? 'translate-x-4' : 'translate-x-0'}`} />
                      </button>
                    </div>
                    {/* Orientation toggle */}
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-400 font-mono uppercase">View</span>
                      <button
                        onClick={() => setIsLandscape(l => !l)}
                        className="flex items-center gap-1 bg-white/10 hover:bg-white/20 active:scale-95 border border-white/20 px-2 py-1 rounded-lg text-[10px] font-bold text-white transition-all uppercase"
                      >
                        {isLandscape ? <Monitor className="w-3 h-3" /> : <Smartphone className="w-3 h-3" />}
                        {isLandscape ? 'Land' : 'Port'}
                      </button>
                    </div>
                  </div>

                  {/* Record controls */}
                  <div className={`relative flex items-center ${isLandscape ? 'flex-col gap-3' : 'gap-6'}`}>
                     {isRecording && (
                        <div className={`bg-red-500/20 border border-red-500/50 text-red-200 px-3 py-1 rounded-full text-xs font-bold tracking-wider flex items-center gap-1.5 ${isPaused ? 'opacity-60' : 'animate-pulse'} ${isLandscape ? '' : 'absolute -top-12'}`}>
                           <div className={`w-1.5 h-1.5 rounded-full ${isPaused ? 'bg-amber-400' : 'bg-red-500'}`} />
                           {isPaused ? 'PAUSED' : 'REC'} · {formatDuration(recordingSeconds)}
                        </div>
                     )}
                     <button
                       onClick={() => { if (scrollContainerRef.current) scrollContainerRef.current.scrollTop = 0; }}
                       className="p-3 rounded-full bg-slate-800 hover:bg-slate-700 active:scale-90 text-white transition-all shadow-lg border border-white/10 group"
                       title="Restart Script"
                     >
                       <RefreshCw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-500" />
                     </button>
                     <button
                       onClick={stopRecording}
                       className="p-3 rounded-full bg-slate-800 hover:bg-slate-700 text-white transition-all shadow-lg border border-white/10 group"
                       title="Stop & Review"
                       disabled={!isRecording && recordedChunksRef.current.length === 0}
                     >
                       <Square className="w-4 h-4 fill-current group-hover:scale-110 transition-transform" />
                     </button>
                     <button
                       onClick={() => { if (isRecording) { togglePauseRecording(); } else { startRecording(); } }}
                       className={`rounded-full flex items-center justify-center transition-all shadow-xl border-4 ${isLandscape ? 'w-14 h-14' : 'w-20 h-20'} ${
                         isRecording
                           ? (isPaused ? 'bg-amber-500 border-amber-600 scale-95' : 'bg-white border-slate-200')
                           : 'bg-red-600 border-red-700 hover:scale-105'
                       }`}
                     >
                       {isRecording
                         ? (isPaused ? <Play className={`text-white fill-current ml-1 ${isLandscape ? 'w-5 h-5' : 'w-8 h-8'}`} /> : <Pause className={`text-slate-900 fill-current ${isLandscape ? 'w-5 h-5' : 'w-8 h-8'}`} />)
                         : <Circle className={`text-white fill-current ${isLandscape ? 'w-5 h-5' : 'w-8 h-8'}`} />
                       }
                     </button>
                  </div>
                </div>
              </>
            )}

            {mode !== AppMode.TELEPROMPTER && (
              <div className="flex-1 flex items-center justify-center w-full relative">
                  <div className="relative w-[300px] h-[300px] md:w-[500px] md:h-[500px]">
                    <AudioVisualizer analyser={outputAnalyser} color={mode === AppMode.COPILOT ? "#a855f7" : "#60a5fa"} mode="circle" />
                    <div className="absolute inset-0 opacity-40 scale-75">
                       <AudioVisualizer analyser={inputAnalyser} color={mode === AppMode.COPILOT ? "#e879f9" : "#c084fc"} mode="circle" />
                    </div>
                    
                    {mode === AppMode.COPILOT ? (
                       <div className="absolute inset-0 flex items-center justify-center z-10">
                         <div className="bg-black/60 backdrop-blur-xl border border-purple-500/30 rounded-3xl p-8 w-full max-w-md text-center shadow-2xl">
                           <p className="text-xs text-purple-400 uppercase tracking-wider mb-2 font-bold">Suggested Answer</p>
                           <p className="text-2xl md:text-3xl font-bold text-white leading-tight">
                             {currentOutputText || "Listening for questions..."}
                           </p>
                         </div>
                       </div>
                    ) : (
                      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-slate-900 rounded-full border-4 border-slate-800 shadow-2xl flex items-center justify-center z-10">
                         <div className={`w-24 h-24 rounded-full bg-gradient-to-tr from-blue-500 to-cyan-400 ${currentOutputText ? 'animate-pulse' : 'opacity-50'}`} />
                      </div>
                    )}
                  </div>
                  
                  {mode === AppMode.CONVERSATION && (
                     <div className="absolute bottom-8 w-full max-w-2xl px-4 flex flex-col gap-4">
                        {currentOutputText && (
                          <div className="bg-blue-600/20 backdrop-blur-md px-6 py-4 rounded-2xl border border-blue-500/20 shadow-lg text-center">
                             <p className="text-white text-lg font-medium leading-relaxed">{currentOutputText}</p>
                          </div>
                        )}
                     </div>
                  )}
              </div>
            )}

            {mode !== AppMode.TELEPROMPTER && (
              <div className={`bg-slate-900/80 backdrop-blur-2xl border border-white/10 shadow-2xl flex items-center z-50 ${
                isLandscape
                  ? 'absolute right-3 top-1/2 -translate-y-1/2 flex-col rounded-3xl px-3 py-4 gap-4'
                  : 'w-full max-w-fit mx-auto rounded-full px-4 sm:px-8 py-3 sm:py-4 gap-4 sm:gap-8 mb-6 sm:mb-8'
              }`}>
                <button
                  onClick={toggleMic}
                  className={`p-3 rounded-full transition-all active:scale-90 ${isMicOn ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'}`}
                  title="Toggle Microphone"
                >
                  {isMicOn ? <Mic className={isLandscape ? 'w-5 h-5' : 'w-6 h-6'} /> : <MicOff className={isLandscape ? 'w-5 h-5' : 'w-6 h-6'} />}
                </button>

                <button
                  onClick={() => setIsAiMuted(!isAiMuted)}
                  className={`p-3 rounded-full transition-all active:scale-90 ${!isAiMuted ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30'}`}
                >
                  <span className="font-bold text-xs">{isAiMuted ? "UNMUTE\nAI" : "MUTE\nAI"}</span>
                </button>

                <div className={isLandscape ? 'w-8 h-px bg-white/10' : 'w-px h-10 bg-white/10'} />

                <button
                  onClick={() => cleanup(false)}
                  className="p-4 bg-red-600 hover:bg-red-500 active:scale-90 rounded-full text-white shadow-lg shadow-red-900/40 hover:scale-105 transition-all"
                >
                  <PhoneOff className={isLandscape ? 'w-5 h-5' : 'w-6 h-6'} />
                </button>

                <div className={isLandscape ? 'w-8 h-px bg-white/10' : 'w-px h-10 bg-white/10'} />

                <button
                  onClick={toggleCamera}
                  className={`p-3 rounded-full transition-all active:scale-90 ${isCameraOn ? 'bg-slate-700 text-white hover:bg-slate-600' : 'bg-red-500/20 text-red-400 hover:bg-red-500/30'}`}
                  title="Toggle Camera"
                >
                  {isCameraOn ? <Video className={isLandscape ? 'w-5 h-5' : 'w-6 h-6'} /> : <VideoOff className={isLandscape ? 'w-5 h-5' : 'w-6 h-6'} />}
                </button>
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
};

export default LiveSession;