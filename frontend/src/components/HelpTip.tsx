interface HelpTipProps {
  /** Stable DOM id for aria-describedby linkage. */
  id: string;
  /** Tooltip body text. */
  text: string;
}

/**
 * Pure CSS-only tooltip popover. Visible on hover (mouse) and focus-within
 * (keyboard). No new dependencies; no portal or floating-UI machinery.
 */
export function HelpTip({ id, text }: HelpTipProps) {
  return (
    <span className="help-tip">
      <button
        type="button"
        className="help-tip__trigger"
        aria-describedby={id}
        tabIndex={0}
      >
        ?
      </button>
      <span role="tooltip" id={id} className="help-tip__bubble">
        {text}
      </span>
    </span>
  );
}