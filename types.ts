export enum ConnectionState {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  ERROR = 'ERROR',
}

export enum AppMode {
  CONVERSATION = 'CONVERSATION',
  COPILOT = 'COPILOT',
  TELEPROMPTER = 'TELEPROMPTER',
}

export interface MessageLog {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

export interface AudioVisualizerProps {
  analyser: AnalyserNode | null;
  color?: string;
  mode?: 'bar' | 'circle';
}