export interface NavItem {
  label: string;
  to: string;
}

/** Logged-out: Home | Analyze X-Ray | About | Log In | Sign Up */
export const PUBLIC_LINKS: NavItem[] = [
  { label: 'Home', to: '/' },
  { label: 'Analyze X-Ray', to: '/analyze' },
  { label: 'About', to: '/about' },
];

/** Logged-in: Home | Analyze X-Ray | Recent Analyses | About | Profile | Logout */
export const MEMBER_LINKS: NavItem[] = [
  { label: 'Home', to: '/' },
  { label: 'Analyze X-Ray', to: '/analyze' },
  { label: 'Recent Analyses', to: '/history' },
  { label: 'About', to: '/about' },
  { label: 'Profile', to: '/profile' },
];
