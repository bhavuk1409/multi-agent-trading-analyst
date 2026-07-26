import { motion, AnimatePresence } from 'framer-motion';
import type { AgentState } from '../types';
import { NeuralOrb } from './NeuralOrb';

interface AgentNeuralNetProps {
  agents: AgentState[];
  isRunning: boolean;
  ticker: string;
}

const AGENT_COLORS = ['#ffffff', '#bbbbbb', '#888888', '#555555'];

export function AgentNeuralNet({ agents, isRunning, ticker }: AgentNeuralNetProps) {
  const allDone = agents.every(a => a.status === 'done');
  const anyRunning = agents.some(a => a.status === 'running');

  return (
    <div className="neural-net">
      {/* Connection lines SVG removed for clean floating orbital style */}
      <svg
        className="neural-net__lines"
        viewBox="-180 -180 360 360"
        preserveAspectRatio="xMidYMid meet"
      />

      {/* Center nucleus */}
      <div className="neural-net__nucleus">
        <AnimatePresence mode="wait">
          {anyRunning ? (
            <motion.div
              key="running"
              className="nucleus nucleus--running"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
            >
              <motion.div
                className="nucleus__pulse"
                animate={{ scale: [1, 1.6, 1], opacity: [0.4, 0, 0.4] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <span className="nucleus__ticker">{ticker}</span>
              <span className="nucleus__label faint">ANALYZING</span>
            </motion.div>
          ) : allDone ? (
            <motion.div
              key="done"
              className="nucleus nucleus--done"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            >
              <span className="nucleus__ticker">{ticker}</span>
              <span className="nucleus__label" style={{ color: 'rgba(255,255,255,0.6)' }}>COMPLETE</span>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              className="nucleus nucleus--idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <span className="nucleus__ticker nucleus__ticker--idle">{ticker || '—'}</span>
              <span className="nucleus__label faint">READY</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Agent orbs */}
      {agents.map((agent, i) => (
        <NeuralOrb
          key={agent.id}
          agent={agent}
          index={i}
          total={agents.length}
          isActive={agent.status === 'running'}
        />
      ))}
    </div>
  );
}
