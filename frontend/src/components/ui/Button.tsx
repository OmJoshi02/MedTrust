import { forwardRef, type AnchorHTMLAttributes, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { Link, type LinkProps } from 'react-router-dom';
import { buttonStyles, type ButtonSize, type ButtonVariant } from './buttonStyles';
import { Spinner } from './Spinner';

interface StyleProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, StyleProps {
  loading?: boolean;
  leadingIcon?: ReactNode;
  trailingIcon?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant,
    size,
    loading = false,
    disabled,
    className,
    children,
    leadingIcon,
    trailingIcon,
    type = 'button',
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={buttonStyles({ variant, size, className })}
      {...rest}
    >
      {loading ? <Spinner /> : leadingIcon}
      {children}
      {!loading && trailingIcon}
    </button>
  );
});

interface ButtonLinkProps extends LinkProps, StyleProps {
  leadingIcon?: ReactNode;
  trailingIcon?: ReactNode;
}

/** Router link that looks like a button. */
export function ButtonLink({
  variant,
  size,
  className,
  children,
  leadingIcon,
  trailingIcon,
  ...rest
}: ButtonLinkProps) {
  return (
    <Link className={buttonStyles({ variant, size, className })} {...rest}>
      {leadingIcon}
      {children}
      {trailingIcon}
    </Link>
  );
}

interface ButtonAnchorProps extends AnchorHTMLAttributes<HTMLAnchorElement>, StyleProps {
  trailingIcon?: ReactNode;
}

/** Plain anchor (in-page hash links, external URLs) styled as a button. */
export function ButtonAnchor({ variant, size, className, children, trailingIcon, ...rest }: ButtonAnchorProps) {
  return (
    <a className={buttonStyles({ variant, size, className })} {...rest}>
      {children}
      {trailingIcon}
    </a>
  );
}
