"""Headless local LOCC figures from caller-cropped geometry only.

No model loading, slicing, branch selection, recovery, or coordinate conversion.
See plotting_inventory.md in the task output for the existing helpers reused.
"""
import ast
from functools import lru_cache
from math import ceil, sqrt
from pathlib import Path
import sys
from textwrap import fill

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gds_project.config import get_font_properties

_REUSED = [
    'scripts/06_visualization/plot_scanline_layouts.py',
    'scripts/99_experiments/orthogonal_audit_report.py',
    'scripts/04_structure_recognition/generate_profile_normals.py',
    'gds_project/config.py',
]


@lru_cache(maxsize=1)
def _paper_helpers():
    """Load only audited font definitions, literal style and exact axis method.

    Executing the original plotting modules would import GUI/geometry engines or
    run data IO. No import, main, or other method from those modules is executed.
    """
    def source(relative):
        return ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))

    layout = source(_REUSED[0])
    fonts = [node for node in layout.body if isinstance(node, ast.Assign)
             and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id in ('font_en', 'font_en_13')]
    audit = source(_REUSED[1])
    render = next(node for node in audit.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'render_figures')
    style_call = next(node.value for node in render.body
                      if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                      and ast.unparse(node.value.func) == 'plt.rcParams.update')
    style = ast.literal_eval(style_call.args[0])
    profile = source(_REUSED[2])
    plotter = next(node for node in profile.body
                   if isinstance(node, ast.ClassDef) and node.name == 'ArcPlotter')
    equal_axes = next(node for node in plotter.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'set_axes_equal')
    # Keep the method, including @staticmethod, unchanged in an otherwise empty
    # class. Its only external dependency is NumPy; no solver code is evaluated.
    plotter.body = [equal_axes]
    selected = ast.fix_missing_locations(ast.Module(body=fonts + [plotter], type_ignores=[]))
    namespace = {'np': np, 'FontProperties': FontProperties, '__name__': __name__}
    exec(compile(selected, '<existing plotting helpers>', 'exec'), namespace)
    en = namespace['font_en']
    en.set_family([*en.get_family(), 'DejaVu Sans'])
    heading = namespace['font_en_13']
    heading.set_family(en.get_family())
    style.update({'font.family': en.get_family(), 'axes.unicode_minus': False})
    return style, en, heading, get_font_properties(size=13), namespace['ArcPlotter'].set_axes_equal


def _label_font(text, heading, chinese):
    return chinese if any('\u3400' <= char <= '\u9fff' for char in text) else heading


def _array(value, tail, name):
    array = np.asarray(value, dtype=float)
    if not array.size:
        return np.empty((0, *tail))
    if array.ndim != len(tail) + 1 or array.shape[1:] != tail:
        raise ValueError(f'{name} must have shape (N, {", ".join(map(str, tail))})')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite coordinates')
    return array


def _lines(ax, segments, color, label, width=1., **kwargs):
    if len(segments):
        ax.add_collection(LineCollection(segments, colors=color, linewidths=width,
                                         label=label, **kwargs))


def _clip_xyz_lines(lines, bounds):
    """Display-only exact segment/AABB clipping; never change solver geometry."""
    lines = np.asarray(lines, dtype=float).reshape(-1, 2, 3)
    if bounds is None or not len(lines):
        return lines
    start, delta = lines[:, 0], lines[:, 1]-lines[:, 0]
    lo, hi = np.zeros(len(lines)), np.ones(len(lines))
    valid = np.ones(len(lines), dtype=bool)
    for axis in range(3):
        moving = abs(delta[:, axis]) > 1e-14
        valid &= moving | ((start[:, axis] >= bounds[axis, 0]) & (start[:, axis] <= bounds[axis, 1]))
        a = np.divide(bounds[axis, 0]-start[:, axis], delta[:, axis], out=np.full(len(lines), -np.inf), where=moving)
        b = np.divide(bounds[axis, 1]-start[:, axis], delta[:, axis], out=np.full(len(lines), np.inf), where=moving)
        lo = np.maximum(lo, np.minimum(a, b))
        hi = np.minimum(hi, np.maximum(a, b))
    valid &= lo <= hi
    return np.stack((start[valid]+lo[valid, None]*delta[valid], start[valid]+hi[valid, None]*delta[valid]), axis=1)


def _clip_xyz_faces(vertices, faces, bounds):
    """Clip actual triangle polygons to the display box, keeping their planes."""
    triangles = vertices[faces]
    if bounds is None or not len(triangles):
        return list(triangles)
    intersects = np.all(triangles.max(axis=1) >= bounds[:, 0], axis=1) & np.all(triangles.min(axis=1) <= bounds[:, 1], axis=1)
    polygons = []
    for triangle in triangles[intersects]:
        polygon = list(triangle)
        for axis in range(3):
            for boundary, lower in ((bounds[axis, 0], True), (bounds[axis, 1], False)):
                if not polygon:
                    break
                clipped = []
                previous = polygon[-1]
                before = previous[axis] >= boundary if lower else previous[axis] <= boundary
                for current in polygon:
                    inside = current[axis] >= boundary if lower else current[axis] <= boundary
                    if inside != before:
                        alpha = (boundary-previous[axis])/(current[axis]-previous[axis])
                        clipped.append(previous+alpha*(current-previous))
                    if inside:
                        clipped.append(current)
                    previous, before = current, inside
                polygon = clipped
        if len(polygon) >= 3:
            polygons.append(np.asarray(polygon))
    return polygons


def _save(fig, output_path, dpi):
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix('.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Audit export conventions, with the caller's DPI instead of its fixed 160.
    extra = fig.get_default_bbox_extra_artists()
    for ax in fig.axes:
        if hasattr(ax, 'zaxis'):
            extra.extend([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label])
    fig.savefig(path, dpi=dpi, bbox_inches='tight', bbox_extra_artists=extra,
                facecolor='white')
    return str(path)


def plot_local_review(case: dict, output_path, dpi=300) -> dict:
    """Render A: target, B: neighbors, C: horizontal branches, D: actual XYZ.

    Required case fields: candidate_id, s, lower_uz, upper_uz, inferred_curve
    (Nx2), inferred_xyz (Nx3), target_lines_uz (Mx2x2), neighbor_profiles
    ({s, lines_uz, lines_xyz}), horizontal_profiles
    ({z, lines_su, lines_xyz, selected_u}), mesh_vertices (Nx3), mesh_faces
    (Mx3 indices), red_groups (list of Nx2), title_note. Lists and arrays work;
    empty geometry stays empty. Optional truth_curve is labeled hidden truth
    for masking reviews. Optional bounds u/z/s affect only the 2D display.

    Every supplied profile/branch is drawn; the caller owns the five-neighbor
    selection and adaptive/global levels. Returns path and actual reused modules.
    """
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError('dpi must be positive and finite')
    target = _array(case['target_lines_uz'], (2, 2), 'target_lines_uz')
    endpoints = _array([case['lower_uz'], case['upper_uz']], (2,), 'gap endpoints')
    inferred = _array(case['inferred_curve'], (2,), 'inferred_curve')
    inferred_xyz = _array(case['inferred_xyz'], (3,), 'inferred_xyz')
    truth = _array(case.get('truth_curve', []), (2,), 'truth_curve')
    red = [_array(group, (2,), 'red_groups') for group in case['red_groups']]
    neighbors = [(float(p['s']), _array(p['lines_uz'], (2, 2), 'neighbor lines_uz'),
                  _array(p['lines_xyz'], (2, 3), 'neighbor lines_xyz'))
                 for p in case['neighbor_profiles']]
    horizontal = [(float(p['z']), _array(p['lines_su'], (2, 2), 'horizontal lines_su'),
                   _array(p['lines_xyz'], (2, 3), 'horizontal lines_xyz'), p['selected_u'])
                  for p in case['horizontal_profiles']]
    vertices = _array(case['mesh_vertices'], (3,), 'mesh_vertices')
    faces = _array(case['mesh_faces'], (3,), 'mesh_faces')
    if faces.size and (np.any(faces != np.floor(faces))
                       or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('mesh_faces must be integer indices into mesh_vertices')
    faces = faces.astype(np.intp)
    s = float(case['s'])
    is_track = case.get('review_kind') == 'track'
    xyz_bounds = None
    if 'bounds_xyz' in case:
        xyz_bounds = np.asarray(case['bounds_xyz'], dtype=float)
        if xyz_bounds.shape != (3, 2) or not np.isfinite(xyz_bounds).all() or np.any(xyz_bounds[:, 1] <= xyz_bounds[:, 0]):
            raise ValueError('bounds_xyz must be finite ordered XYZ bounds with shape (3, 2)')
        center = xyz_bounds.mean(axis=1)
        radius = np.max(np.diff(xyz_bounds, axis=1))/2
        xyz_bounds = np.column_stack((center-radius, center+radius))
    if not np.isfinite(s) or any(not np.isfinite(p[0]) for p in neighbors + horizontal):
        raise ValueError('profile positions must be finite')
    if any(p[3] is not None and not np.isfinite(p[3]) for p in horizontal):
        raise ValueError('selected_u must be finite or None')
    style, font, heading, chinese, equal_axes = _paper_helpers()
    neighbor_colors = [plt.get_cmap('tab10')(i % 10) for i in range(len(neighbors))]
    horizontal_colors = [plt.get_cmap('Dark2')(i % 8) for i in range(len(horizontal))]

    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 10), layout='constrained')
        try:
            a, b, c = [fig.add_subplot(2, 2, i) for i in (1, 2, 3)]
            d = fig.add_subplot(2, 2, 4, projection='3d')
            _lines(a, target, '#737d8d', 'Target fragments', 1.4)
            _lines(a, [group for group in red if len(group)], '#c73845', 'Baseline red groups', 2.)
            for point, marker, color, label in zip(
                    [] if is_track else endpoints, ('^', 'v'), ('#177eaa', '#284c64'),
                    ('Lower gap endpoint', 'Upper gap endpoint')):
                a.scatter(*point, marker=marker, c=color, s=55, edgecolors='white',
                          linewidths=.6, label=label, zorder=6)
            if len(truth):
                a.plot(*truth.T, color='#8b5aa5', linestyle=':', linewidth=2.2,
                       marker='.' if len(truth) == 1 else None,
                       label='Hidden truth (masking only)', zorder=4)
            if len(inferred):
                a.plot(*inferred.T, color='#e48a22', linestyle='--', linewidth=2.,
                       marker='.' if len(inferred) == 1 else None, label='Inferred curve', zorder=5)
            else:
                a.text(.02, .98, 'Track association only' if is_track else 'No inferred curve supplied', transform=a.transAxes,
                       va='top', fontsize=9)

            for (position, uz, _), color in zip(neighbors, neighbor_colors):
                _lines(b, uz, color, f's = {position:.6g} m', 1., alpha=.8)
            _lines(b, target, '#25313d', f'Target s = {s:.6g} m', 1.8)
            if len(inferred):
                b.plot(*inferred.T, color='#e48a22', linestyle='--', linewidth=2.,
                       marker='.' if len(inferred) == 1 else None, label='Inferred curve')

            # Heights below are used only to name caller-supplied levels; never
            # compute sections or substitute synthetic horizontal observations.
            lower_z, upper_z = endpoints[:, 1]
            levels = lower_z + (upper_z - lower_z) * np.array([.25, .5, .75])
            selected_label = 'Selected branch (supplied)'
            for (z, su, _, selected_u), color in zip(horizontal, horizontal_colors):
                matched = np.flatnonzero(np.isclose(z, levels, rtol=0,
                                                   atol=max(abs(upper_z - lower_z)*1e-7, 1e-9)))
                prefix = f'z{(25, 50, 75)[matched[0]]}' if len(matched) else 'Extra level'
                _lines(c, su, color, f'{prefix}: z = {z:.6g} m', 1.25)
                if selected_u is not None:
                    c.scatter(s, selected_u, s=50, c=[color], edgecolors='black',
                              linewidths=.8, zorder=5, label=selected_label)
                    selected_label = '_nolegend_'
            c.axvline(s, color='#25313d', linestyle='--', linewidth=1.,
                      label=f'Target s_i = {s:.6g} m')
            if not horizontal:
                c.text(.02, .98, 'No horizontal profiles supplied', transform=c.transAxes, va='top')

            bounds = case.get('bounds', {})
            for ax, title, xkey, ykey in (
                    (a, 'A | Track association review window' if is_track else 'A | Target fragments and gap', 'u', 'z'),
                    (b, f'B | Neighbor profiles ({len(neighbors)} supplied)', 'u', 'z'),
                    (c, 'C | Horizontal profiles - all branches', 's', 'u')):
                ax.autoscale()
                ax.margins(.06)
                if xkey in bounds:
                    ax.set_xlim(bounds[xkey])
                if ykey in bounds:
                    ax.set_ylim(bounds[ykey])
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel(f'{xkey} (m)', fontproperties=font)
                ax.set_ylabel(f'{ykey} (m)', fontproperties=font)
                ax.set_title(title, fontproperties=heading, loc='left')
                ax.ticklabel_format(useOffset=False, style='plain')
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=9)

            xyz_parts = [vertices, inferred_xyz]
            legend = []
            if len(faces):
                d.add_collection3d(Poly3DCollection(_clip_xyz_faces(vertices, faces, xyz_bounds), facecolors='#9aa6b2',
                                                    edgecolors='none', alpha=.18))
                legend.append(Patch(facecolor='#9aa6b2', alpha=.4, label='Local mesh'))
            elif len(vertices):
                d.scatter(*vertices.T, s=2, c='#9aa6b2', alpha=.4)
                legend.append(Line2D([], [], color='#9aa6b2', marker='.',
                                     linestyle='none', label='Supplied mesh vertices (no faces)'))
            for (position, _, xyz), color in zip(neighbors, neighbor_colors):
                xyz = _clip_xyz_lines(xyz, xyz_bounds)
                if len(xyz):
                    d.add_collection3d(Line3DCollection(xyz, colors=color, linewidths=1.))
                    xyz_parts.append(xyz.reshape(-1, 3))
                    legend.append(Line2D([], [], color=color, label=f'V: s = {position:.6g} m'))
            for (z, _, xyz, _), color in zip(horizontal, horizontal_colors):
                xyz = _clip_xyz_lines(xyz, xyz_bounds)
                if len(xyz):
                    d.add_collection3d(Line3DCollection(xyz, colors=color, linewidths=1., linestyles=':'))
                    xyz_parts.append(xyz.reshape(-1, 3))
                    legend.append(Line2D([], [], color=color, linestyle=':', label=f'H: z = {z:.6g} m'))
            if len(inferred_xyz):
                d.plot(*inferred_xyz.T, color='#e48a22', linestyle='--', linewidth=2.5,
                       marker='.' if len(inferred_xyz) == 1 else None, label='Inferred curve')
                legend.append(Line2D([], [], color='#e48a22', linestyle='--', label='Inferred curve'))
            xyz_parts = [points for points in xyz_parts if len(points)]
            if xyz_parts:
                xyz = np.concatenate(xyz_parts)
                d.auto_scale_xyz(*xyz.T)
            else:
                d.text2D(.02, .95, 'No supplied XYZ geometry', transform=d.transAxes)
            if xyz_bounds is not None:
                d.set_xlim(xyz_bounds[0])
                d.set_ylim(xyz_bounds[1])
                d.set_zlim(xyz_bounds[2])
            equal_axes(d)
            d.set_box_aspect((1, 1, 1))
            d.view_init(elev=24, azim=-55)
            d.set_xlabel('X (m)', fontproperties=font)
            d.set_ylabel('Y (m)', fontproperties=font)
            d.set_zlabel('Z (m)', fontproperties=font)
            d.ticklabel_format(useOffset=False, style='plain')
            for axis in (d.xaxis, d.yaxis, d.zaxis):
                axis.set_major_locator(MaxNLocator(nbins=3))
            d.set_title('D | Local mesh and profiles (actual XYZ)', fontproperties=heading, loc='left')
            if legend:
                d.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, -.06),
                         ncol=3, fontsize=8)
            title = f"{case['candidate_id']} | s = {s:.6g} m"
            if case['title_note']:
                title += '\n' + fill(str(case['title_note']), width=110)
            fig.suptitle(title, fontproperties=_label_font(title, heading, chinese))
            path = _save(fig, output_path, dpi)
        finally:
            plt.close(fig)
    return {'path': path, 'modules_reused': list(_REUSED)}


def plot_montage(items: list[dict], output_path, title: str) -> str:
    """Save up to 20 caller-selected raster figures with ID/selection/label.

    No sampling, repetition or truncation of cases. Empty input produces a
    labeled empty sheet. Thumbnails limit the grid to roughly 5100 x 5000 pixels
    at 20 cases; full-resolution individual figures remain separate artifacts.
    """
    if len(items) > 20:
        raise ValueError('plot_montage accepts at most 20 caller-selected items')
    style, _, heading, chinese, _ = _paper_helpers()
    columns = min(4, max(1, ceil(sqrt(len(items)))))
    rows = max(1, ceil(len(items) / columns))
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, figsize=(7*columns, 5.4*rows),
                                 squeeze=False, layout='constrained')
        try:
            for ax in axes.flat:
                ax.set_axis_off()
            for ax, item in zip(axes.flat, items):
                with Image.open(item['path']) as source:
                    source.thumbnail((1260, 900), Image.Resampling.LANCZOS)
                    ax.imshow(np.asarray(source.convert('RGB')), interpolation='nearest')
                caption = str(item['candidate_id'])
                if item.get('selection'):
                    caption += f" | {item['selection']}"
                if item.get('label'):
                    caption += '\n' + fill(str(item['label']), width=65)
                ax.set_title(caption, fontproperties=_label_font(caption, heading, chinese), loc='left')
            if not items:
                axes[0, 0].text(.5, .5, 'No supplied cases', ha='center', va='center',
                                transform=axes[0, 0].transAxes, fontproperties=heading)
            fig.suptitle(title, fontproperties=_label_font(title, heading, chinese))
            return _save(fig, output_path, 180)
        finally:
            plt.close(fig)
