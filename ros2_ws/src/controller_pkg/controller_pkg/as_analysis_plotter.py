#!/usr/bin/env python3
"""Plot AS-module diagnostics: per-expert prediction errors and expert selection probabilities.

This script auto-discovers or accepts explicit numpy files and produces PNGs.
It is robust to the logger writing placeholders (empty arrays) and will exit
gracefully if no usable data is available.
"""
import argparse
import glob
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def make_save_dir(base_path=None):
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "mpc_logs")
    os.makedirs(base_path, exist_ok=True)
    return base_path


def try_load(path):
    if path is None:
        return None
    return np.load(path, allow_pickle=True)


def auto_find(pattern):
    search_dir = os.path.join(os.path.dirname(__file__), "mpc_logs")
    matches = sorted(glob.glob(os.path.join(search_dir, pattern)))
    return matches[-1] if matches else None


def normalize_errors(errors):
    """Coerce various error-array shapes into a (T, P) numpy array.

    Returns None if errors is empty or cannot be coerced.
    """
    if errors is None:
        return None
    # Try numeric coercion first
    try:
        a = np.asarray(errors)
    except Exception:
        a = np.asarray(errors, dtype=object)

    if a.size == 0:
        return None

    # If it's a numeric array already
    if np.issubdtype(a.dtype, np.number):
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim == 2:
            return a
        # higher dims not supported
        return None

    # Object arrays: try stack/list-handling
    try:
        seq = list(a)
        # If each element is an array-like of shape (T,) or (T,1) or (T,P)
        stacked = np.stack([np.asarray(x) for x in seq])
    except Exception:
        # last-resort: flatten then coerce
        flat = np.asarray(seq)
        if flat.ndim == 1:
            return flat.reshape(-1, 1)
        if flat.ndim == 2:
            return flat
        return None

    # stacked is numeric
    if stacked.ndim == 2:
        s0, s1 = stacked.shape
        # Heuristic: if stacked is (P, T) convert to (T, P)
        if s0 <= 10 and s1 > s0:
            return stacked.T
        return stacked
    if stacked.ndim == 3:
        # maybe (P, T, 1) or (T, P, 1). squeeze last dim
        if stacked.shape[-1] == 1:
            squeezed = stacked.squeeze(-1)
            if squeezed.ndim == 2:
                s0, s1 = squeezed.shape
                if s0 <= 10 and s1 > s0:
                    return squeezed.T
                return squeezed
    return None


def compute_errors_from_preds(preds, gt):
    """Compute per-expert per-sample Euclidean errors.

    preds: can be shaped (P, T, 3) or (T, P, 3) or list-of-arrays where each entry is (T,3)
    gt: shaped (T,3)
    Returns: errors (T, P)
    """
    preds = np.asarray(preds, dtype=object)
    # If object array from ragged save, try to stack
    if preds.dtype == object:
        try:
            stacked = np.stack([p for p in preds])
            preds = stacked
        except Exception:
            preds = np.asarray(preds.tolist())

    if preds.ndim == 3:
        p0, p1, p2 = preds.shape
        if p0 <= 10 and p1 == gt.shape[0] and p2 == 3:
            # (P, T, 3)
            P = p0
            T = p1
            errors = np.zeros((T, P))
            for i in range(P):
                errors[:, i] = np.linalg.norm(preds[i] - gt, axis=1)
            return errors
        elif p0 == gt.shape[0] and p2 == 3:
            # (T, P, 3)
            T = p0
            P = p1
            errors = np.zeros((T, P))
            for i in range(P):
                errors[:, i] = np.linalg.norm(preds[:, i, :] - gt, axis=1)
            return errors

    raise ValueError('Unsupported preds shape for computing errors. Expected (P,T,3) or (T,P,3) or list-of-(T,3).')


def plot_expert_errors(errors, save_path):
    errors = normalize_errors(errors)
    if errors is None:
        raise RuntimeError('Could not coerce errors into a (T,P) numeric array.')
    T, P = errors.shape
    plt.figure(figsize=(9, 4))
    for p in range(P):
        plt.plot(np.arange(T), errors[:, p], label=f'Expert {p}')
    plt.xlabel('Sample index')
    plt.ylabel('Position error (m)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_expert_probs(probs, save_path):
    if probs is None:
        return
    p = np.asarray(probs)
    if p.size == 0:
        return
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    T, P = p.shape
    plt.figure(figsize=(9, 4))
    xs = np.arange(T)
    ys = p.T
    labels = [f'Expert {i}' for i in range(P)]
    plt.stackplot(xs, *ys, labels=labels)
    plt.xlabel('Sample index')
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='AS module analysis plotter')
    parser.add_argument('--gt', help='Ground truth trajectory .npy file (T x 3)')
    parser.add_argument('--preds', help='Expert predictions .npy file (P x T x 3 or T x P x 3 or list)')
    parser.add_argument('--errors', help='Per-expert errors .npy file (T x P)')
    parser.add_argument('--probs', help='Expert probability trace .npy file (T x P)')
    parser.add_argument('--selected', help='Selected expert index trace .npy file (T,)')
    parser.add_argument('--label', default='as_analysis')
    args = parser.parse_args()

    # Auto-discover files if not provided
    if args.gt is None:
        args.gt = auto_find('*_gt_*.npy')
    if args.preds is None:
        args.preds = auto_find('*_expert_preds_*.npy')
    if args.errors is None:
        args.errors = auto_find('*_expert_errors_*.npy')
    if args.probs is None:
        args.probs = auto_find('*_expert_probs_*.npy')
    if args.selected is None:
        args.selected = auto_find('*_expert_selected_*.npy')

    # Report which files we will use (or attempted to auto-find)
    print('File selection:')
    print('  gt     ->', args.gt)
    print('  preds  ->', args.preds)
    print('  errors ->', args.errors)
    print('  probs  ->', args.probs)
    print('  selected ->', args.selected)

    if args.gt is None:
        print('Ground truth file not found. Provide --gt or place GT file in mpc_logs.')
        return

    gt = try_load(args.gt)
    gt = np.asarray(gt)
    if gt.ndim == 1:
        # try to reshape flat to (T,3) when possible
        if gt.size % 3 == 0:
            gt = gt.reshape(-1, 3)
        else:
            gt = gt.reshape(-1, 3)

    save_dir = make_save_dir()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    errors = None
    if args.errors is not None:
        e = try_load(args.errors)
        errors = normalize_errors(e)

    # If errors not available, try compute from preds
    if errors is None and args.preds is not None:
        preds = try_load(args.preds)
        try:
            errors = compute_errors_from_preds(preds, gt)
        except Exception as ex:
            print('Could not compute errors from preds:', ex)
            errors = None

    if errors is None:
        # Print diagnostic summary to help user/debug logger
        def info(path):
            if path is None:
                return 'MISSING'
            try:
                a = try_load(path)
                if a is None:
                    return 'MISSING'
                return f'dtype={getattr(a,"dtype",None)} ndim={getattr(a,"ndim",None)} shape={getattr(a,"shape",None)} size={getattr(a,"size",None)}'
            except Exception as ex:
                return f'FAILED_LOAD({ex})'

        print('\nDiagnostics:')
        print('  gt:    ', info(args.gt))
        print('  preds: ', info(args.preds))
        print('  errors:', info(args.errors))
        print('  probs: ', info(args.probs))
        print('  sel:   ', info(args.selected))
        print('\nNo usable per-expert error data found. Provide --errors or --preds and valid GT. Exiting.')
        return

    # plot expert errors
    fname_err = os.path.join(save_dir, f"{args.label}_expert_errors_{ts}.png")
    plot_expert_errors(errors, fname_err)
    print(f"Saved expert errors plot: {fname_err}")

    # plot expert probabilities if provided
    if args.probs is not None:
        probs = try_load(args.probs)
        fname_probs = os.path.join(save_dir, f"{args.label}_expert_probs_{ts}.png")
        plot_expert_probs(probs, fname_probs)
        print(f"Saved expert probs plot: {fname_probs}")

    # if selected trace provided, also plot which expert was selected over time
    if args.selected is not None:
        sel = try_load(args.selected)
        sel = np.asarray(sel)
        if sel.size > 0:
            fname_sel = os.path.join(save_dir, f"{args.label}_expert_selected_{ts}.png")
            plt.figure(figsize=(9, 2))
            plt.plot(np.arange(sel.shape[0]), sel, drawstyle='steps-post')
            plt.yticks(np.unique(sel))
            plt.xlabel('Sample index')
            plt.ylabel('Selected expert')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fname_sel, dpi=200)
            plt.close()
            print(f"Saved selected-expert plot: {fname_sel}")


if __name__ == '__main__':
    main()
