// Light formatting for answers: `code`, **bold** and [n] citation markers.
// Built from React elements (no innerHTML), so model output can never inject HTML.
import { Fragment } from "react";

const TOKEN = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[\d+\])/g;

export function FormattedText({ text }: { text: string }) {
  return (
    <>
      {text.split(TOKEN).map((part, i) => {
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={i}>{part.slice(1, -1)}</code>;
        if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (/^\[\d+\]$/.test(part)) return <sup key={i} className="cite-mark">{part.slice(1, -1)}</sup>;
        return <Fragment key={i}>{part}</Fragment>;
      })}
    </>
  );
}
