import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Ticker } from '../types';

interface CompanySelectProps {
  selected: Ticker;
  onChange: (ticker: Ticker) => void;
  disabled?: boolean;
}

interface CompanyDetails {
  ticker: Ticker;
  name: string;
  exchange: string;
  status: string;
  meta: string;
}

const COMPANIES: Record<Ticker, CompanyDetails> = {
  AAPL: { ticker: 'AAPL', name: 'Apple', exchange: 'NASDAQ', status: 'ACTIVE', meta: 'Tech · USD' },
  GOOGL: { ticker: 'GOOGL', name: 'Alphabet', exchange: 'NASDAQ', status: 'ACTIVE', meta: 'Tech · USD' },
  MSFT: { ticker: 'MSFT', name: 'Microsoft', exchange: 'NASDAQ', status: 'ACTIVE', meta: 'Tech · USD' },
  TSLA: { ticker: 'TSLA', name: 'Tesla', exchange: 'NASDAQ', status: 'ACTIVE', meta: 'Auto · USD' },
  NVDA: { ticker: 'NVDA', name: 'NVIDIA', exchange: 'NASDAQ', status: 'ACTIVE', meta: 'Chips · USD' },
};

export function CompanySelect({ selected, onChange, disabled }: CompanySelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const current = COMPANIES[selected];

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="company-select" ref={containerRef}>
      <button
        type="button"
        className={`company-select__trigger ${isOpen ? 'company-select__trigger--open' : ''}`}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
      >
        <div className="company-select__left">
          <div className="company-select__ticker-mark" aria-hidden="true">
            {current.ticker}
          </div>

          <span className="company-select__name">{current.name}</span>
        </div>

        <div className="company-select__right">
          <span className="company-select__meta">{current.meta}</span>
          <span className="company-select__arrow" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}>▼</span>
        </div>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="company-select__dropdown"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
          >
            {Object.values(COMPANIES).map((comp) => (
              <button
                key={comp.ticker}
                type="button"
                className={`company-select__option ${selected === comp.ticker ? 'company-select__option--selected' : ''}`}
                onClick={() => {
                  onChange(comp.ticker);
                  setIsOpen(false);
                }}
              >
                <div className="company-select__left">
                  <div
                    className="company-select__ticker-mark"
                    aria-hidden="true"
                  >
                    {comp.ticker}
                  </div>
                  <span className="company-select__name">{comp.name}</span>
                </div>
                <div className="company-select__right">
                  <span className="company-select__meta">{comp.meta}</span>
                </div>
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
