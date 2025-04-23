# Listing allometry
# https://www.wrighters.io/use-ipywidgets-with-jupyter-notebooks/
# https://towardsdatascience.com/interactive-controls-for-jupyter-notebooks-f5c94829aee6
# https://stackoverflow.com/questions/60998665/is-it-possible-to-make-another-ipywidgets-widget-appear-based-on-dropdown-select

# creating class for UI selecting scenario ex-ante
# Listing allometry
# for interactivity
# for interactivity
import ipywidgets as widgets
import pandas as pd
from IPython.display import display, clear_output
import traceback # Import for detailed error printing

from ipywidgets import interact, interact_manual, interactive
from google.colab import output
output.enable_custom_widget_manager()

from .utils import filter_or_selection


# Assuming pandas is imported as pd elsewhere

class SelectingScenario(widgets.VBox):
    def __init__(
        self, allometric_column_filter: pd.DataFrame, name_column_species_allo: str = ""
    ):
        self.allometric_column_filter = allometric_column_filter
        self.name_column_species_allo = name_column_species_allo

        # Select the literature formula - country of origin
        self.country_allometry = widgets.SelectMultiple(
            options=list(self.allometric_column_filter["Country of Use"].unique()),
            value=["Indonesia", "Singapore"], # Example default
            description="Country of Use:",
            disabled=False,
        )

        # --- Placeholder for Allometry Type ---
        self.allometry_select = widgets.SelectMultiple(
            options=[], value=[], description="Allom.type:", disabled=True
        )
        # --- Placeholder for Species Selection Area ---
        # This VBox will hold checkboxes and the zone-specific widgets
        self.species_selection_area = widgets.VBox([])
        # Initially hide the species area
        self.species_selection_area.layout.display = 'none'

        # --- Store zone widgets ---
        self.species_widgets_by_zone = {} # Will store {'zone': VBox(SelectMultiple, Output)}
        self.zone_checkboxes = {}         # Will store {'zone': Checkbox}

        self.df_tree_selected = {
            "production_zone": pd.DataFrame(),
            "protected_zone": pd.DataFrame(),
        }

        # Observe country selection
        self.country_allometry.observe(self._update_allo_options, names=["value"])
        # Observe allometry selection
        self.allometry_select.observe(self._update_species_widgets, names=["value"])

        # Initial layout
        children = [
            self.country_allometry,
            self.allometry_select,
            self.species_selection_area # Add the placeholder here
        ]
        super().__init__(children=children)

        # Trigger initial population based on default country
        self._update_allo_options({'new': self.country_allometry.value})


    # Refactored filter function using .isin() for clarity and safety
    def filter_dataframe(self, df, column_name, values):
         if not isinstance(values, (list, tuple)):
             values = [values] # Ensure it's a list for isin
         if not values: # Handle empty selection
             # Return an empty DataFrame with the same columns
             return df.iloc[0:0]
         return df[df[column_name].isin(values)]

    # Renamed for clarity: This updates Allometry options based on Country
    def _update_allo_options(self, change):
        country_selected = change["new"]
        if not country_selected:
             self.allometry_select.options = []
             self.allometry_select.value = []
             self.allometry_select.disabled = True
             self.species_selection_area.layout.display = 'none' # Hide species area
             # Clear downstream selections/data if country is deselected
             self._clear_species_widgets_and_data()
             return # Stop processing if no country selected

        self.country_filtered_df = self.filter_dataframe(
             self.allometric_column_filter, "Country of Use", country_selected
        )

        new_options = list(self.country_filtered_df["Allometric Formula, Type"].unique())
        # Keep existing selection if possible, otherwise select first
        current_value = self.allometry_select.value
        new_value = [v for v in current_value if v in new_options]
        if not new_value and new_options:
            new_value = [new_options[0]] # Default to first if previous selection invalid

        self.allometry_select.options = new_options
        self.allometry_select.value = new_value # Triggers its observer
        self.allometry_select.disabled = not bool(new_options)

        # If allometry options become empty, hide species area
        if not new_options:
             self.species_selection_area.layout.display = 'none'
             self._clear_species_widgets_and_data()
        # If allometry has options but no value automatically set, trigger update manually
        elif new_value and not self.allometry_select.value:
             self._update_species_widgets({'new': new_value})


    def _clear_species_widgets_and_data(self):
        """Helper to clear species widgets and reset data."""
        self.species_selection_area.children = [] # Clear all children (checkboxes, zone VBoxes)
        self.species_widgets_by_zone = {}
        self.zone_checkboxes = {}
        self.df_tree_selected = {
            "production_zone": pd.DataFrame(),
            "protected_zone": pd.DataFrame(),
        }

    # Renamed for clarity: This updates Species widgets based on Allometry Type
    def _update_species_widgets(self, change):
        allo_type_selected = change["new"]

        # Clear previous species widgets and data first
        self._clear_species_widgets_and_data()

        if not self.name_column_species_allo or not allo_type_selected or not hasattr(self, 'country_filtered_df'):
            self.species_selection_area.layout.display = 'none' # Hide if no species column or selection
            return

        # Filter based on currently selected country AND allometry type
        self.filtered_by_country_allo_type = self.filter_dataframe(
            self.country_filtered_df,
            "Allometric Formula, Type",
            allo_type_selected,
        )

        if self.filtered_by_country_allo_type.empty:
            self.species_selection_area.layout.display = 'none' # Hide if filter results in no species
            return

        # Prepare species list for options
        combine_list = [
            f"{idx}-{species}"
            for idx, species in zip(
                self.filtered_by_country_allo_type.index,
                self.filtered_by_country_allo_type[self.name_column_species_allo],
            )
        ]

        if not combine_list: # If no species found for this combo
            self.species_selection_area.layout.display = 'none'
            return

        # --- Setup widgets for each zone ---
        zone_widget_children = [] # Holds checkboxes and the zone VBoxes

        zones = ["production_zone", "protected_zone"] # Define your zones
        output_widgets = {} # Temporary holder within this scope

        def create_filter_function(zone, species_select_widget, output_widget):
            # Use a closure to capture the correct widgets for each zone
            def filter_function(species_selection_wid):
                selected_indices = [int(i.split("-")[0]) for i in species_selection_wid]
                if selected_indices:
                    self.df_tree_selected[zone] = self.filtered_by_country_allo_type.loc[
                        selected_indices
                    ].copy() # Use .copy() to avoid SettingWithCopyWarning
                    self.df_tree_selected[zone]["zone"] = zone
                else:
                    self.df_tree_selected[zone] = pd.DataFrame()

                # Ensure output_widget is cleared and updated
                output_widget.clear_output(wait=True) # Use wait=True for smoother updates
                with output_widget:
                    if not self.df_tree_selected[zone].empty:
                        print(f"Selected for {zone}:") # Add label for clarity
                        display(self.df_tree_selected[zone])
                    # else:
                    #     print(f"No species selected for {zone}.") # Optional: message when empty
            return filter_function

        for zone in zones:
            output_widgets[zone] = widgets.Output()
            species_select_widget = widgets.SelectMultiple(
                options=combine_list,
                value=[],
                description=f'Select species in {zone.replace("_", " ")}: ',
                disabled=False,
                rows=min(len(combine_list), 10), # Limit rows for better display
                layout=widgets.Layout(width="auto", min_width="400px") # Adjust layout
            )

            # Create the VBox containing the selector and its output area
            zone_vbox = widgets.VBox([species_select_widget, output_widgets[zone]])
            # Hide this zone's widgets initially
            zone_vbox.layout.display = 'none'
            self.species_widgets_by_zone[zone] = zone_vbox

            # Create the checkbox for this zone
            checkbox = widgets.Checkbox(
                value=False,
                description=f'Configure {zone.replace("_", " ").capitalize()}',
                disabled=False,
                indent=False,
            )
            self.zone_checkboxes[zone] = checkbox

            # Define the checkbox handler using a closure to capture zone
            def make_checkbox_observer(current_zone):
                def on_checkbox_change(change):
                    target_widget = self.species_widgets_by_zone[current_zone]
                    if change["new"]: # Checked
                        target_widget.layout.display = 'flex' # Show the VBox for this zone
                        # --- Crucially, set up the interactive link *here* ---
                        # Check if interactive instance already exists for this widget
                        if not hasattr(species_select_widget, '_interactive_instance'):
                           species_select_widget._interactive_instance = interactive(
                                create_filter_function(current_zone, species_select_widget, output_widgets[current_zone]),
                                species_selection_wid=species_select_widget,
                           )
                           # We don't need to display the result of interactive() itself,
                           # as the widgets are already in our layout.
                    else: # Unchecked
                        target_widget.layout.display = 'none' # Hide the VBox
                        # Reset selection and data when hiding
                        species_select_widget.value = []
                        self.df_tree_selected[current_zone] = pd.DataFrame()
                        output_widgets[current_zone].clear_output()
                return on_checkbox_change

            checkbox.observe(make_checkbox_observer(zone), names="value")

            # Add checkbox and the (initially hidden) zone VBox to the list
            zone_widget_children.append(checkbox)
            zone_widget_children.append(zone_vbox)


        # Update the species area container with the new structure
        self.species_selection_area.children = tuple(zone_widget_children)
        # Make the whole species area visible now that it's populated
        self.species_selection_area.layout.display = 'block'


    def select_species(self, name_column_species_allo=None):
        """Ensures species column name is set and returns current selection."""
        if name_column_species_allo is not None:
            self.name_column_species_allo = name_column_species_allo
            # If name is set/changed, we might need to re-trigger the species update
            # based on the current allometry selection.
            current_allo_value = self.allometry_select.value
            if current_allo_value:
                 self._update_species_widgets({'new': current_allo_value})

        return self.df_tree_selected

    @property
    def selected_data(self):
        """Returns a dictionary of selected DataFrames for each zone."""
        # Combine data from both zones if needed, or return as is
        # Example: Concatenate non-empty dataframes
        # all_selected = pd.concat([df for df in self.df_tree_selected.values() if not df.empty], ignore_index=True)
        # return all_selected
        # Or just return the dictionary:
        return {zone: df.copy() for zone, df in self.df_tree_selected.items()} # Return copies


# --- Example Usage (Illustrative) ---
# Assuming you have your allometry_df DataFrame loaded
# manager = InterfaceManager(...) # Or however you create SelectingScenario
# display(manager.wm) # If wm is the SelectingScenario instance

class Project_Setting_Species(widgets.VBox):
    def __init__(self, species_name, grouping_max_year):
        self.species_name = species_name
        self.grouping_max_year = grouping_max_year  # Make it an instance variable

        self.caption = widgets.Label(
            value="Please edit the following: harvesting cycle, natural thinning, and how frequently manual thinning should occur!"
        )
        self.wid_species_name = widgets.HTML(
            value=f"<b><font color='black'>{species_name}</b>"
        )

        self.year_harvesting = self.create_int_text_widget(
            "harvest_cycle_year:", self.grouping_max_year[self.species_name]
        )
        self.mortality_percent = self.create_int_text_widget(
            "mortality_rate_percent:", 20
        )
        self.natural_thinning = self.create_int_text_widget(
            "natural_thinning_percent:", 5
        )
        self.frequency_thinning = self.create_int_text_widget(
            "how many times manual thinning:", 0
        )

        self.wrapper_species_harvest = widgets.HBox(
            [
                self.wid_species_name,
                self.year_harvesting,
                self.mortality_percent,
                self.natural_thinning,
                self.frequency_thinning,
            ]
        )

        self.list_widget_holder_params = widgets.HBox()
        self.list_widget_holder_params.add_class("box2_style")

        self.year_harvesting.observe(self.check_range_harvest, names=["value"])
        self.mortality_percent.observe(self.check_mortality_input, names=["value"])
        self.natural_thinning.observe(self.check_range_thinning, names=["value"])
        self.frequency_thinning.observe(self.add_thinning_cycle, names=["value"])

        super().__init__(
            children=[
                self.caption,
                self.wrapper_species_harvest,
                self.list_widget_holder_params,
            ]
        )

    @staticmethod
    def create_int_text_widget(description, value):
        return widgets.IntText(
            description=description, style={"description_width": "initial"}, value=value
        )

    def check_range_harvest(self, change):
        cycle_year_selected = change["new"]
        max_year = self.grouping_max_year[self.species_name]

        if cycle_year_selected > max_year:
            self.caption.value = f"The harvesting cycle exceeds the available growth rate data (max year {max_year}). Please lower it or add more data."
        elif cycle_year_selected <= 5:
            self.caption.value = "Harvesting cycle should be more than 5 years!"
        else:
            self.caption.value = (
                f"Harvesting cycle of {cycle_year_selected} years seems reasonable!"
            )

    def check_mortality_input(self, change):
        mortality_percent = change["new"]
        if mortality_percent > 30:
            self.caption.value = f"Are you sure the mortality rate is more than {mortality_percent}%? This will significantly reduce the population in the first year!"
        else:
            self.caption.value = f"You set mortality rate: {mortality_percent}%, okay!"

    def check_range_thinning(self, change):
        natural_thinning_percent = change["new"]
        if natural_thinning_percent > 10:
            self.caption.value = f"Are you sure the natural thinning rate is {natural_thinning_percent}%? This will reduce the population every year!"
        else:
            self.caption.value = (
                f"You set natural thinning rate: {natural_thinning_percent}%, okay!"
            )

    def add_thinning_cycle(self, change):
        num_thinning_freq = change["new"]
        new_widgets = []
        for i in range(num_thinning_freq):
            year_thinning = self.create_int_text_widget(f"thinning_cycle{i+1}_year:", 0)
            percent_thinning = self.create_int_text_widget(
                f"thinning_cycle{i+1}_percent:", 0
            )
            new_widgets.extend([year_thinning, percent_thinning])  # Add both widgets

        self.list_widget_holder_params.children = tuple(new_widgets)

    @property
    def check(self):
        return {w.description: w.value for w in self.list_widget_holder_params.children}


class CSUEntryForm:
    def __init__(self, csu_seedling):
        """
        Initialize the data entry form based on the structure of the provided DataFrame.
        Sets default values for widgets based on the column names and data types.
        """
        self.csu_seedling = csu_seedling
        self.original_columns = csu_seedling.columns  # Store the original columns
        self.original_dtypes = csu_seedling.dtypes  # Store the original data types
        # The DataFrame to which rows will be added
        self.widgets_dict = {}  # Dictionary to hold widgets for each column
        self.output = widgets.Output()  # Output widget to display the DataFrame

        # Initialize widgets based on DataFrame columns
        self._initialize_widgets()

        # Button to add a new row
        self.add_row_button = widgets.Button(description="Add Row")
        self.add_row_button.on_click(self.add_row_to_df)

        # Button to reset the form
        self.reset_button = widgets.Button(description="Reset Form")
        self.reset_button.on_click(self.reset_form)

        # Layout the form
        form_items = list(self.widgets_dict.values()) + [
            self.add_row_button,
            self.reset_button,
        ]
        self.form = widgets.VBox(form_items)

    def _initialize_widgets(self):
        """
        Create widgets dynamically based on DataFrame column names and data types.
        Set default values where specified.
        """
        for col_name, col_dtype in self.csu_seedling.dtypes.items():
            if col_name == "Plot_ID":
                self.widgets_dict[col_name] = widgets.IntText(
                    value=1, description=col_name
                )  # Default Plot_ID = 1
            elif col_name == "Plot_Name":
                self.widgets_dict[col_name] = widgets.Text(
                    value="general", description=col_name
                )  # Default Plot_Name = 'general'
            elif col_name == "zone":
                self.widgets_dict[col_name] = widgets.Dropdown(
                    options=["production_zone", "protected_zone"],
                    value="protected_zone",
                    description=col_name,
                )  # Default zone
            elif col_name == "area_ha":
                self.widgets_dict[col_name] = widgets.FloatText(
                    value=50.0, description=col_name
                )  # Default area_ha = 50

            elif col_name == "is_replanting":
                self.widgets_dict[col_name] = widgets.Checkbox(
                    value=False, description=col_name
                )  
            elif col_name == "year_start":
                self.widgets_dict[col_name] = widgets.IntText(
                    value=1, description=col_name
                )  # Default start_year = 1
            elif col_name == "mu":
                self.widgets_dict[col_name] = widgets.FloatText(
                    value=1.0, description=col_name
                )  # Default mu = 1
            else:
                # For other species columns, default to 1000
                self.widgets_dict[col_name] = widgets.IntText(
                    value=1000, description=col_name
                )

    def add_row_to_df(self, button):
        """
        Add a new row to the DataFrame based on the current widget values.
        """
        # Create a dictionary to store values for the new row
        new_row = {
            col_name: widget.value for col_name, widget in self.widgets_dict.items()
        }

        # Convert the new row to a DataFrame
        new_row_df = pd.DataFrame([new_row])

        # Append the new row DataFrame to the existing DataFrame using pd.concat
        self.csu_seedling = pd.concat(
            [self.csu_seedling, new_row_df], ignore_index=True
        )

        # Display the updated DataFrame
        with self.output:
            self.output.clear_output()
            display(self.csu_seedling)

        # Update the Plot_ID widget to the next sequential number
        max_plot_id = self.csu_seedling["Plot_ID"].max()  # Get the current max Plot_ID
        self.widgets_dict["Plot_ID"].value = max_plot_id + 1  # Set to the next number

    def reset_form(self, button):
        """
        Reset all widgets to their default values. and empty the df
        """
        # Clear all rows by setting the DataFrame to an empty DataFrame with the original columns
        self.csu_seedling = pd.DataFrame(columns=self.original_columns).astype(
            self.original_dtypes
        )
        # Set Plot_ID to the next sequential number in the existing DataFrame
        if "Plot_ID" in self.csu_seedling.columns:
            max_plot_id = self.csu_seedling["Plot_ID"].max()
            self.widgets_dict["Plot_ID"].value = (
                max_plot_id + 1 if not pd.isna(max_plot_id) else 1
            )
        else:
            self.widgets_dict["Plot_ID"].value = 1

        # Reset other fields to default values
        for col_name, widget in self.widgets_dict.items():
            if col_name == "Plot_ID":
                continue  # Plot_ID is already handled
            elif col_name == "Plot_Name":
                widget.value = "general"
            elif col_name == "zone":
                widget.value = "production_zone"
            elif col_name == "area_ha":
                widget.value = 50.0
            elif col_name == "year_start":
                widget.value = 1
            elif col_name == "mu":
                widget.value = 1.0
            else:
                widget.value = 1000  # Reset species count columns to 1000

        # Display the cleared DataFrame
        with self.output:
            self.output.clear_output()

    def display_form(self):
        """
        Display the form and the output area.
        """
        display(self.form, self.output)
