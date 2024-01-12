from circles import fit_circles

line_width = 2
small_circles = True
dst = 'data/analysis'
src = 'data/combined'
center_pickle = f'data/centers.p'

fit_circles(analysis_path=dst, data_path=src, centers_path=center_pickle, lw=line_width, small=small_circles)
