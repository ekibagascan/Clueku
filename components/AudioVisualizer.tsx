import React, { useEffect, useRef } from 'react';
import { AudioVisualizerProps } from '../types';

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ analyser, color = '#3b82f6', mode = 'circle' }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !analyser) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // High DPI scaling
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    let animationId: number;

    const draw = () => {
      animationId = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, rect.width, rect.height);

      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      if (mode === 'circle') {
        const barWidth = (rect.width / bufferLength) * 2.5;
        let barHeight;

        ctx.beginPath();
        for (let i = 0; i < bufferLength; i++) {
          barHeight = dataArray[i] / 2;
          
          // Only draw lower frequencies for a cleaner look
          if (i > bufferLength / 3) break; 

          const angle = (i / (bufferLength / 3)) * Math.PI * 2;
          const radius = 50 + barHeight * 0.5;
          
          const xPos = centerX + Math.cos(angle) * radius;
          const yPos = centerY + Math.sin(angle) * radius;

          if (i === 0) {
            ctx.moveTo(xPos, yPos);
          } else {
            ctx.lineTo(xPos, yPos);
          }
        }
        ctx.closePath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Inner glow
        ctx.beginPath();
        ctx.arc(centerX, centerY, 40 + (dataArray[0] / 10), 0, Math.PI * 2);
        ctx.fillStyle = `${color}33`; // Transparent hex
        ctx.fill();
      } else {
        // Bar mode (Waveform-ish)
        const sliceWidth = rect.width * 1.0 / bufferLength;
        let x = 0;

        ctx.beginPath();
        ctx.moveTo(0, centerY);
        
        for(let i = 0; i < bufferLength; i++) {
          const v = dataArray[i] / 128.0;
          const y = v * centerY; // amplify

          // Mirror effect for waveform
          if (i === 0) ctx.moveTo(x, (rect.height - y + centerY) / 2);
          else ctx.lineTo(x, (rect.height - y + centerY) / 2);

          x += sliceWidth;
        }
        ctx.lineTo(rect.width, centerY);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [analyser, color, mode]);

  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full absolute top-0 left-0 pointer-events-none"
    />
  );
};

export default AudioVisualizer;