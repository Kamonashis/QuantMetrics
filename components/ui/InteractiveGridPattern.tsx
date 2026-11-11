'use client';

import React, { useState } from 'react';

interface InteractiveGridPatternProps {
  width?: number;
  height?: number;
  squares?: [number, number]; // [horizontal, vertical]
  className?: string;
  squaresClassName?: string;
}

export function InteractiveGridPattern({
  width = 32,
  height = 32,
  squares = [24, 12],
  className = '',
  squaresClassName = '',
}: InteractiveGridPatternProps) {
  const [hoveredSquare, setHoveredSquare] = useState<number | null>(null);
  const [horizontal, vertical] = squares;
  const totalSquares = horizontal * vertical;

  return (
    <svg
      width={width * horizontal}
      height={height * vertical}
      className={`pointer-events-auto absolute inset-0 h-full w-full stroke-zinc-300/30 dark:stroke-zinc-800/40 select-none ${className}`}
      aria-hidden="true"
    >
      <defs>
        <pattern
          id="quant-grid-pattern"
          width={width}
          height={height}
          patternUnits="userSpaceOnUse"
          x={-1}
          y={-1}
        >
          <path
            d={`M ${width} 0 L 0 0 0 ${height}`}
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray="2 2"
          />
        </pattern>
      </defs>

      <rect width="100%" height="100%" fill="url(#quant-grid-pattern)" />

      <svg x={-1} y={-1} className="overflow-visible">
        {Array.from({ length: totalSquares }).map((_, index) => {
          const x = (index % horizontal) * width;
          const y = Math.floor(index / horizontal) * height;
          const isHovered = hoveredSquare === index;

          return (
            <rect
              key={index}
              x={x}
              y={y}
              width={width}
              height={height}
              className={`stroke-0 transition-all duration-300 ${
                isHovered
                  ? 'fill-gold-500/15 dark:fill-gold-400/20'
                  : 'fill-transparent'
              } ${squaresClassName}`}
              onMouseEnter={() => setHoveredSquare(index)}
              onMouseLeave={() => setHoveredSquare(null)}
            />
          );
        })}
      </svg>
    </svg>
  );
}
