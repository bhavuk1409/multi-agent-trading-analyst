import { motion } from 'framer-motion';
import type { NewsArticle } from '../types';
import { IconChevronRight } from './Icons';

interface NewsFeedProps {
  articles: NewsArticle[];
}

/** First letter of the source domain, used as the leading glyph on each row. */
function sourceInitial(source: string): string {
  const cleaned = source.replace(/^(https?:\/\/)?(www\.)?/, '').trim();
  return (cleaned[0] ?? '?').toUpperCase();
}

/** Pull a clean domain out of a `source` field that may include a path. */
function sourceDomain(source: string): string {
  return source.replace(/^(https?:\/\/)?(www\.)?/, '').split('/')[0];
}

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
          const domain = sourceDomain(article.source);
          const initial = sourceInitial(domain);

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
              <div className="news-item__source-mark" aria-hidden="true">
                {initial}
              </div>
              <div className="news-item__body">
                <div className="news-item__title">{article.title}</div>
                <div className="news-item__meta">
                  <span className="news-item__source">{domain}</span>
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
