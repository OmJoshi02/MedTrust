import type { DiseaseName } from '@/types/analysis';
import { mulberry32 } from './seed';

/**
 * Abstract, monochrome SAMPLE imagery used only in demo mode.
 * These are drawn shapes, not real radiographs, and carry a visible label.
 */

const SIZE = 512;

function radiographLayers(seed: number): string {
  const random = mulberry32(seed);
  const jitter = (amount: number) => (random() - 0.5) * amount;

  const ribs: string[] = [];
  for (let i = 0; i < 9; i += 1) {
    const y = 120 + i * 36;
    const bow = 18 + i * 3;
    ribs.push(
      `<path d="M256 ${y - 6} C 200 ${y - bow} 130 ${y - 4 + i * 2} 78 ${y + 28 + i * 4}" />`,
      `<path d="M256 ${y - 6} C 312 ${y - bow} 382 ${y - 4 + i * 2} 434 ${y + 28 + i * 4}" />`,
    );
  }

  return `
    <rect width="${SIZE}" height="${SIZE}" fill="url(#bg)" />
    <ellipse cx="${170 + jitter(8)}" cy="${262 + jitter(8)}" rx="104" ry="172" fill="url(#lung)" />
    <ellipse cx="${342 + jitter(8)}" cy="${262 + jitter(8)}" rx="104" ry="172" fill="url(#lung)" />
    <g filter="url(#soft)" fill="#cfcfcf">
      <ellipse cx="256" cy="262" rx="40" ry="190" fill-opacity="0.22" />
      <ellipse cx="${288 + jitter(10)}" cy="${338 + jitter(10)}" rx="72" ry="86" fill-opacity="0.32" />
      <rect x="244" y="60" width="24" height="400" fill-opacity="0.18" />
    </g>
    <g filter="url(#fine)" fill="none" stroke="#d9d9d9" stroke-opacity="0.16" stroke-width="5" stroke-linecap="round">
      ${ribs.join('\n      ')}
      <path d="M256 96 C 210 84 150 96 96 130" stroke-opacity="0.22" />
      <path d="M256 96 C 302 84 362 96 416 130" stroke-opacity="0.22" />
    </g>
    <g filter="url(#soft)" fill="#e6e6e6" fill-opacity="0.2">
      <ellipse cx="150" cy="452" rx="110" ry="30" />
      <ellipse cx="364" cy="452" rx="110" ry="30" />
    </g>
    <rect width="${SIZE}" height="${SIZE}" fill="url(#vignette)" />
    <rect width="${SIZE}" height="${SIZE}" filter="url(#grain)" opacity="0.14" />
  `;
}

function defs(seed: number): string {
  return `
    <defs>
      <radialGradient id="bg" cx="50%" cy="42%" r="78%">
        <stop offset="0" stop-color="#4a4a4a" />
        <stop offset="1" stop-color="#0c0c0c" />
      </radialGradient>
      <radialGradient id="lung" cx="50%" cy="50%" r="50%">
        <stop offset="0" stop-color="#7a7a7a" stop-opacity="0.6" />
        <stop offset="0.72" stop-color="#5a5a5a" stop-opacity="0.34" />
        <stop offset="1" stop-color="#2a2a2a" stop-opacity="0" />
      </radialGradient>
      <radialGradient id="vignette" cx="50%" cy="50%" r="72%">
        <stop offset="0.6" stop-color="#000" stop-opacity="0" />
        <stop offset="1" stop-color="#000" stop-opacity="0.65" />
      </radialGradient>
      <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="9" /></filter>
      <filter id="fine"><feGaussianBlur stdDeviation="1.6" /></filter>
      <filter id="grain" x="0" y="0" width="100%" height="100%">
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="${seed % 997}" />
        <feColorMatrix type="saturate" values="0" />
      </filter>
    </defs>
  `;
}

function caption(text: string): string {
  return `<text x="18" y="${SIZE - 18}" fill="#ffffff" fill-opacity="0.6" font-family="Arial, Helvetica, sans-serif" font-size="13" letter-spacing="2.4">${text}</text>`;
}

function wrap(inner: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${SIZE} ${SIZE}" width="${SIZE}" height="${SIZE}">${inner}</svg>`;
}

export function sampleRadiographSvg(seed: number): string {
  return wrap(
    `${defs(seed)}${radiographLayers(seed)}${caption('SAMPLE ILLUSTRATION — NOT A REAL RADIOGRAPH')}`,
  );
}

/** Where each finding's sample heat region is drawn. */
const FOCUS: Partial<Record<DiseaseName, Array<[number, number, number]>>> = {
  Cardiomegaly: [[288, 340, 78]],
  Effusion: [[120, 424, 66], [394, 424, 66]],
  Pneumothorax: [[110, 150, 64]],
  Atelectasis: [[130, 360, 56]],
  Consolidation: [[380, 330, 62]],
  Pneumonia: [[372, 300, 64]],
  Edema: [[160, 280, 70], [352, 280, 70]],
  Emphysema: [[150, 220, 84], [362, 220, 84]],
  Fibrosis: [[140, 330, 58], [372, 330, 58]],
  Hernia: [[290, 440, 46]],
  Pleural_Thickening: [[84, 270, 40]],
};

export function sampleHeatmapSvg(seed: number, disease: DiseaseName | undefined, label: string): string {
  const random = mulberry32(seed + 17);
  const spots =
    (disease ? FOCUS[disease] : undefined) ??
    [
      [150 + random() * 60, 200 + random() * 140, 58 + random() * 22],
      [330 + random() * 60, 200 + random() * 140, 52 + random() * 20],
    ];

  const blobs = spots
    .map(
      ([cx, cy, r], index) => `
        <circle cx="${cx}" cy="${cy}" r="${r * 1.7}" fill="url(#heat)" opacity="${0.5 - index * 0.08}" />
        <circle cx="${cx}" cy="${cy}" r="${r * 0.8}" fill="url(#heat)" opacity="0.7" />`,
    )
    .join('');

  return wrap(`
    ${defs(seed)}
    <defs>
      <radialGradient id="heat" cx="50%" cy="50%" r="50%">
        <stop offset="0" stop-color="#ffffff" stop-opacity="0.95" />
        <stop offset="0.45" stop-color="#d6d6d6" stop-opacity="0.55" />
        <stop offset="1" stop-color="#ffffff" stop-opacity="0" />
      </radialGradient>
    </defs>
    ${radiographLayers(seed)}
    <g style="mix-blend-mode:screen">${blobs}</g>
    ${caption(`SAMPLE EXPLANATION — ${label.toUpperCase()}`)}
  `);
}

export function svgBlob(svg: string): Blob {
  return new Blob([svg], { type: 'image/svg+xml' });
}
