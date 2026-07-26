import { motion } from 'framer-motion';
import type { NewsArticle } from '../types';
import { IconChevronRight } from './Icons';

interface NewsFeedProps {
  articles: NewsArticle[];
}

const SENTIMENT_CONFIG = {
  positive: {
    color: 'var(--buy)',
    bg: 'var(--buy-dim)',
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="18,15 12,9 6,15" />
      </svg>
    ),
  },
  negative: {
    color: 'var(--sell)',
    bg: 'var(--sell-dim)',
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="6,9 12,15 18,9" />
      </svg>
    ),
  },
  neutral: {
    color: 'var(--hold)',
    bg: 'var(--hold-dim)',
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg>
    ),
  },
};

export function NewsFeed({ articles }: NewsFeedProps) {
  return (
    <motion.div
      className="news-feed glass"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="news-feed__header">
        <div className="row">
          <div className="dot dot--live" />
          <span className="news-feed__title">LIVE NEWS FEED</span>
        </div>
      </div>

      <div className="news-feed__list">
        {articles.map((article, i) => {
          const cfg = SENTIMENT_CONFIG[article.sentiment] ?? SENTIMENT_CONFIG.neutral;

          return (
            <motion.a
              key={i}
              href={article.url === '#' ? undefined : article.url}
              target={article.url !== '#' ? '_blank' : undefined}
              rel="noopener noreferrer"
              className="news-item"
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08, duration: 0.35 }}
            >
              <div
                className="news-item__sentiment"
                style={{ background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.color}30` }}
              >
                {cfg.icon}
              </div>
              <div className="news-item__body">
                <div className="news-item__title">{article.title}</div>
                <div className="news-item__meta">
                  <span className="news-item__source">{article.source}</span>
                  <span className="faint" style={{ fontSize: 10 }}>·</span>
                  <span className="faint">{article.published_date}</span>
                </div>
              </div>
              <div className="news-item__arrow" style={{ color: 'var(--text-3)' }}>
                <IconChevronRight size={14} color="currentColor" />
              </div>
            </motion.a>
          );
        })}
      </div>
    </motion.div>
  );
}
