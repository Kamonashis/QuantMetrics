import 'react';

declare global {
  interface MathMLElement extends Element {}

  namespace React.JSX {
    interface IntrinsicElements {
      math: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement> & { display?: 'inline' | 'block' }, MathMLElement>;
      mrow: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mi: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement> & { mathvariant?: string }, MathMLElement>;
      mn: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mo: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement> & { movablelimits?: string; stretchy?: string }, MathMLElement>;
      msup: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      msub: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      msubsup: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mfrac: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement> & { linethickness?: string }, MathMLElement>;
      msqrt: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mroot: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mtext: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      munder: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mover: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      munderover: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mtable: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mtr: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
      mtd: React.DetailedHTMLProps<React.HTMLAttributes<MathMLElement>, MathMLElement>;
    }
  }
}
