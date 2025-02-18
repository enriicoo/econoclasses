import matplotlib.pyplot as plt
from typing import Optional, Union, Literal, List, Tuple
from microeconomics.preferences import UtilityCurve

class Plotter:
    """
    Class for managing and displaying multiple plots.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        plots: List[UtilityCurve],
        title: str,
        subtitles: Optional[List[str]] = None,
        cmap: Union[None, str, List[str]] = None,
        label: Union[bool, List[bool]] = False,
        label_position: Optional[Literal['first', 'mid', 'last']] = None,
        label_direction: Optional[Literal['dynamic', 'horizontal', 'vertical']] = 'dynamic',
        show_colorbar: Union[bool, List[bool]] = False,
        show_equation: Union[bool, List[bool]] = False,
        show_prices: Union[bool, List[bool]] = False,
        show_optima: Union[bool, List[bool]] = False,
        show_restriction: Union[bool, List[bool]] = False,
        box_position: Union[Tuple[str, str], List[Tuple[str, str]]] = ('low','right')
    ):
        """
        Initializes the Plotter with a grid of plots.

        :param rows: Number of rows in the plot grid.
        :param cols: Number of columns in the plot grid.
        :param plots: List of plot objects to display.
        :param title: Main title for the entire plot grid.
        :param subtitles: List of subtitles for each subplot.
        :param cmap: Colormap or list of colormaps for the plots. If None, 'viridis' is used.
        :param label: If True, labels are added to the plots. Could be a single bool or a list of bools.
        :param label_position: Position of the labels ('first','mid','last'). Could be single or list.
        :param label_direction: Direction of the labels ('dynamic','horizontal','vertical'). Single or list.
        :param show_colorbar: If True, color bars are displayed. Single or list of bools.
        :param show_equation: If True, shows the function equation as the subplot title. Single or list.
        :param show_prices: If True, a small legend-like box with "Px=?? / Py=??" is added on the plot.
        :param box_position: A tuple or list of tuples indicating ("low"/"high", "left"/"right").
        """
        self.rows = rows
        self.cols = cols
        self.plots = plots
        self.title = title
        self.subtitles = subtitles

        self.fig, self.axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        self.fig.suptitle(title, fontsize=16)

        n_plots = len(plots)

        # Resolve everything
        self.cmap = self._resolve_param(cmap, 'viridis', n_plots)
        self.label = self._resolve_param(label, False, n_plots)
        self.label_position = self._resolve_param(label_position, 'mid', n_plots)
        self.label_direction = self._resolve_param(label_direction, 'dynamic', n_plots)
        self.show_colorbar = self._resolve_param(show_colorbar, False, n_plots)
        self.show_equation = self._resolve_param(show_equation, False, n_plots)
        self.show_prices = self._resolve_param(show_prices, False, n_plots)
        self.show_optima = self._resolve_param(show_optima, False, n_plots)
        self.show_restriction = self._resolve_param(show_restriction, False, n_plots)
        self.box_position = self._resolve_param(box_position, ('low','right'), n_plots)

        for i, plot_obj in enumerate(self.plots):
            ax = self.axes.ravel()[i]

            # Grab the i-th parameter after we've resolved them
            this_cmap = self.cmap[i] or 'viridis'
            this_label = self.label[i]
            this_label_pos = self.label_position[i]
            this_label_dir = self.label_direction[i]
            this_colorbar = self.show_colorbar[i]
            this_equation = self.show_equation[i]
            this_prices = self.show_prices[i]
            this_optima = self.show_optima[i]
            this_boxpos = self.box_position[i]
            this_restriction = self.show_restriction[i]

            # Determine subplot title
            if self.subtitles is not None and i < len(self.subtitles):
                user_subtitle = self.subtitles[i]
            else:
                user_subtitle = None

            if user_subtitle is not None and user_subtitle != '':
                subplot_title = user_subtitle
            elif this_equation and hasattr(plot_obj, 'get_equation_latex'):
                # If show_equation is True and the plot object has an equation
                subplot_title = plot_obj.get_equation_latex()
            else:
                subplot_title = None

            # Automatically call the plot method with parameters
            plot_obj.plot(
                ax=ax,
                cmap=this_cmap,
                label=this_label,
                label_position=this_label_pos,
                label_direction=this_label_dir,
                show_colorbar=this_colorbar,
                show_equation=this_equation,
                title=subplot_title,
                show_prices=this_prices,
                show_optima=this_optima,
                box_position=this_boxpos,
                show_restriction=this_restriction
            )

            # Hide any unused subplots
        total_plots = rows * cols
        if len(plots) < total_plots:
            for j in range(len(plots), total_plots):
                self.axes.ravel()[j].axis('off')

    @staticmethod
    def _resolve_param(param, default, n):
        """
        Returns a list of length n.
          - If param is None/''/[], fill with default for all n.
          - If param is not a list (like single bool/str/tuple), replicate for all n.
          - If param is a list shorter than n, fill the rest with default.
          - If param is a list longer than n, raise an error.
        """
        if param is None or param == '' or (isinstance(param, list) and len(param) == 0):
            return [default] * n
        if not isinstance(param, list):
            # single item
            return [param] * n
        # param is a list
        if len(param) < n:
            # extend it with default
            return param + [default] * (n - len(param))
        elif len(param) > n:
            raise ValueError("Length of list argument exceeds number of plots.")
        else:
            # same length
            return param

    @staticmethod
    def show():
        """
        Displays the plots with tight layout adjustments.
        """
        plt.tight_layout()
        plt.show()
