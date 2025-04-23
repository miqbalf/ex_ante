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
from ipywidgets import interact, interact_manual, interactive
from google.colab import output
output.enable_custom_widget_manager()

from .utils import filter_or_selection



'''
class SelectingScenario(widgets.VBox):
    def __init__(
        self, allometric_column_filter: pd.DataFrame, name_column_species_allo: str = ""
    ):

        self.allometric_column_filter = allometric_column_filter
        self.name_column_species_allo = name_column_species_allo

        # Select the literature formula - country of origin
        self.country_allometry = widgets.SelectMultiple(
            options=list(self.allometric_column_filter["Country of Use"].unique()),
            value=["Indonesia", "Singapore"],
            description="Country of Use:",
            disabled=False,
        )

        self.list_widget_holder = widgets.HBox()
        self.df_tree_selected = {
            "production_zone": pd.DataFrame(),
            "protected_zone": pd.DataFrame(),
        }  # Initialize empty dataframes

        # Placeholder for species selection widget
        self.widget_species_select = None

        # Observe country selection
        self.country_allometry.observe(self._add_allo_type, names=["value"])

        # Add initial widgets
        children = [self.country_allometry, self.list_widget_holder]
        super().__init__(children=children)

    # Function to automate filter selection
    def filter_or_selection(self, df_string_var, column_name, *args):
        filter_df_string = f"{df_string_var}["
        for i in range(len(args)):
            if args[-1] != args[i]:
                filter_df_string += (
                    f'({df_string_var}["{column_name}"] == "{args[i]}" ) |'
                )
            else:
                filter_df_string += (
                    f'({df_string_var}["{column_name}"] == "{args[i]}" )]'
                )
        return eval(filter_df_string)

    def _add_allo_type(self, change):
        print("Triggered _add_allo_type with:", change["new"])
        country_selected = change["new"]
        self.country_filtered_df = self.filter_or_selection(
            "self.allometric_column_filter", "Country of Use", *country_selected
        )

        # Allometry type selection
        self.allometric_select = widgets.SelectMultiple(
            options=list(self.country_filtered_df["Allometric Formula, Type"].unique()),
            value=[
                list(self.country_filtered_df["Allometric Formula, Type"].unique())[0]
            ],
            description="Allom.type:",
            disabled=False,
        )

        # Update list holder with new widgets
        self.list_widget_holder.children = (self.allometric_select,)

        # Add observer for dynamic updates
        self.allometric_select.observe(self.update_species_select, names="value")

        # Initialize or update species selection widget
        self.update_species_select()

    def update_species_select(self, *args):
        """Update the species selection widget based on the selected allometry type."""
        if self.name_column_species_allo:
            # Clear previous widget if it exists
            if self.widget_species_select in self.children:
                self.children = [
                    child
                    for child in self.children
                    if child != self.widget_species_select
                ]

            # Get the updated species selection dictionary
            species_selection_dict = self.setup_species_widgets(
                self.name_column_species_allo
            )

            # Wrap only the checkboxes in a VBox initially
            self.widget_species_select = widgets.VBox(
                [widgets.VBox(list(species_selection_dict["checkboxes"].values()))]
            )

            # Store the species selection and output widgets for each zone separately
            self.species_selection_widgets = {
                zone: widgets.VBox(
                    [
                        species_selection_dict["species_select_widgets"][zone],
                        species_selection_dict["output_widgets"][zone],
                    ]
                )
                for zone in species_selection_dict["species_select_widgets"].keys()
            }

            # Add the checkboxes VBox to the main children
            self.children += (self.widget_species_select,)

    def setup_species_widgets(self, name_column_species_allo):
        """Sets up species selection widgets and handles selection logic."""
        if not hasattr(self, "country_filtered_df"):
            return {
                "checkboxes": {},
                "species_select_widgets": {},
                "output_widgets": {},
            }

        filtered_by_country_allo_type = self.filter_or_selection(
            "self.country_filtered_df",
            "Allometric Formula, Type",
            *self.allometric_select.value,
        )
        combine_list = [
            f"{idx}-{species}"
            for idx, species in zip(
                filtered_by_country_allo_type.index,
                filtered_by_country_allo_type[name_column_species_allo],
            )
        ]

        output_widgets = {
            "production_zone": widgets.Output(),
            "protected_zone": widgets.Output(),
        }

        def filter_function(zone, species_selection_wid):
            selected_indices = [int(i.split("-")[0]) for i in species_selection_wid]
            if selected_indices:  # Only update if there are selected species
                self.df_tree_selected[zone] = filtered_by_country_allo_type.loc[
                    selected_indices
                ]
                self.df_tree_selected[zone]["zone"] = zone
            else:
                self.df_tree_selected[zone] = (
                    pd.DataFrame()
                )  # Reset to an empty DataFrame if no selection

            output_widgets[zone].clear_output()  # Clear output to avoid duplication
            with output_widgets[zone]:
                if not self.df_tree_selected[zone].empty:
                    display(self.df_tree_selected[zone])

        species_select_widgets = {
            zone: widgets.SelectMultiple(
                options=combine_list,
                value=[],  # Start with no selection
                description=f'Selecting species in {zone.replace("_", " ")}: ',
                disabled=False,
                rows=len(combine_list),
                layout=widgets.Layout(width="600px"),
            )
            for zone in output_widgets.keys()
        }

        checkboxes = {
            zone: widgets.Checkbox(
                value=False,
                description=f'Select {zone.replace("_", " ").capitalize()}',
                disabled=False,
                indent=False,
            )
            for zone in output_widgets.keys()
        }

        def on_checkbox_change(change):
            zone = change["owner"].description.split(" ")[1].lower() + "_zone"
            if change["new"]:
                # Checkbox checked - show only the widgets for the selected zone
                if (
                    self.species_selection_widgets[zone]
                    not in self.widget_species_select.children
                ):
                    self.widget_species_select.children += (
                        self.species_selection_widgets[zone],
                    )
                # ✅ Call the function directly for Colab compatibility
                filter_function(zone, species_selection_wid=species_select_widgets[zone])
            else:
                # Checkbox unchecked - reset selection, clear output, and remove widgets for the zone
                species_select_widgets[zone].value = []  # Clear selections
                self.df_tree_selected[zone] = pd.DataFrame()  # Reset to an empty DataFrame
                output_widgets[zone].clear_output()  # Clear the output widget

                # ✅ REMOVE the zone widget instead of adding it again
                self.widget_species_select.children = tuple(
                    child
                    for child in self.widget_species_select.children
                    if child != self.species_selection_widgets[zone]
                )

        for zone, checkbox in checkboxes.items():
            checkbox.observe(on_checkbox_change, names="value")

        # Return dictionary with widgets setup
        return {
            "checkboxes": checkboxes,
            "species_select_widgets": species_select_widgets,
            "output_widgets": output_widgets,
        }

    def select_species(self, name_column_species_allo=None):
        """Sets up or returns the current selection based on available data."""
        # Use the provided or already set column name
        if name_column_species_allo is not None:
            self.name_column_species_allo = name_column_species_allo

        # If no species selection widgets are present, set them up based on the current country and allometry type
        if self.name_column_species_allo and not self.widget_species_select:
            self.update_species_select()

        # Return the current selection as a dictionary
        return self.df_tree_selected

    @property
    def selected_data(self):
        """Returns a dictionary of selected DataFrames for each zone."""
        return {zone: df for zone, df in self.df_tree_selected.items() if not df.empty}
'''

# Assuming filter_or_selection is robust or replaced later

class SelectingScenario(widgets.VBox):
    def __init__(
        self, allometric_column_filter: pd.DataFrame, name_column_species_allo: str = ""
    ):
        print("--- SelectingScenario Init ---") # DEBUG
        self.allometric_column_filter = allometric_column_filter
        self.name_column_species_allo = name_column_species_allo

        # --- Widgets ---
        # Country Selector
        self.country_allometry = widgets.SelectMultiple(
            # Ensure options are valid before setting value
            options=list(self.allometric_column_filter["Country of Use"].unique()),
            value=[c for c in ["Indonesia", "Singapore"] if c in self.allometric_column_filter["Country of Use"].unique()], # Safer default value
            description="Country of Use:",
            disabled=False,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='auto')
        )

        # Holder for Allometry Selector
        self.list_widget_holder = widgets.HBox()

        # *** NEW: Dedicated Output Area for Species Selection Part ***
        self.species_output_area = widgets.Output()
        # ************************************************************

        # --- Data Store ---
        self.df_tree_selected = {
            "production_zone": pd.DataFrame(),
            "protected_zone": pd.DataFrame(),
        }

        # --- Placeholders ---
        self.widget_species_select = None # VBox holding checkboxes and zone widgets
        self.species_selection_widgets = {} # Dict storing VBoxes for each zone's species list+output
        self.allometric_select = None # Placeholder for the allometry widget itself
        self.country_filtered_df = pd.DataFrame() # Placeholder

        # --- Observers ---
        self.country_allometry.observe(self._add_allo_type, names="value")

        # --- Initial Layout ---
        # Children now include the new output area
        initial_children = [self.country_allometry, self.list_widget_holder, self.species_output_area]
        super().__init__(children=initial_children)

        # --- Trigger initial setup ---
        # Call _add_allo_type with the initial value to populate allometry options
        print("Triggering initial _add_allo_type call.") #DEBUG
        self._add_allo_type({'new': self.country_allometry.value, 'owner': self.country_allometry, 'type': 'change', 'name': 'value'})
        print("--- SelectingScenario Init Finished ---") # DEBUG


    # Function to automate filter selection (Keep as is, but consider replacing eval)
    def filter_or_selection(self, df_string_var, column_name, *args):
        # Basic safety checks for eval
        safe_args = [str(arg).replace('"', '\\"') for arg in args if arg]
        base_df = eval(df_string_var) # Get the DataFrame instance

        if not safe_args:
             print(f"WARN: No valid args for filtering {column_name}.") # DEBUG
             return base_df.head(0)

        # --- Alternative using Pandas boolean indexing (safer than eval) ---
        conditions = None
        for arg in safe_args:
            condition = (base_df[column_name] == arg)
            if conditions is None:
                conditions = condition
            else:
                conditions = conditions | condition # Combine with OR

        if conditions is None: # Should not happen if safe_args is not empty, but belt-and-suspenders
             return base_df.head(0)

        print(f"Filtering {df_string_var} where '{column_name}' is in {safe_args}") # DEBUG
        return base_df[conditions]
        # --- End Alternative ---

        # Original eval code (commented out if using alternative)
        # filter_df_string = f"{df_string_var}["
        # conditions_str = []
        # for arg in safe_args:
        #     conditions_str.append(f'({df_string_var}["{column_name}"] == "{arg}")')
        # filter_df_string += " | ".join(conditions_str) + "]"
        # print(f"EVAL FILTER: {filter_df_string}") # DEBUG
        # try:
        #     return eval(filter_df_string)
        # except Exception as e:
        #      print(f"ERROR during eval: {e}")
        #      return base_df.head(0)


    def _add_allo_type(self, change):
        print("--- _add_allo_type ---") # DEBUG
        country_selected = change["new"]
        print(f"Countries selected: {country_selected}") # DEBUG

        # Clear downstream widgets first
        self.list_widget_holder.children = []
        with self.species_output_area:
             clear_output()
        self.widget_species_select = None
        self.species_selection_widgets = {}

        if not country_selected:
             print("No country selected. Clearing allometry and species areas.") # DEBUG
             return # Stop if selection is empty

        try:
            # Use the safer filtering method if implemented
            self.country_filtered_df = self.filter_or_selection(
                "self.allometric_column_filter", "Country of Use", *country_selected
            )
            # Or use original:
            # self.country_filtered_df = self.filter_or_selection_eval(
            #     "self.allometric_column_filter", "Country of Use", *country_selected
            # )
            print(f"Country filtered DF shape: {self.country_filtered_df.shape}") # DEBUG
            if self.country_filtered_df.empty:
                 print("WARN: No data found for selected countries.") # DEBUG
                 return # Stop if no data

            allo_options = sorted(list(self.country_filtered_df["Allometric Formula, Type"].unique())) # Sort options
            print(f"Allometry options generated: {allo_options}") # DEBUG
            if not allo_options:
                 print("WARN: No allometry types found for these countries.") # DEBUG
                 return # Stop if no options

            # Allometry type selection widget
            self.allometric_select = widgets.SelectMultiple(
                options=allo_options,
                value=[allo_options[0]], # Default to first option safely
                description="Allom.type:",
                disabled=False,
                style={'description_width': 'initial'}, layout=widgets.Layout(width='auto')
            )
            print(f"Created allometry selector with default value: {self.allometric_select.value}") #DEBUG

        except Exception as e:
            print(f"ERROR in _add_allo_type filtering/widget creation: {e}") # DEBUG
            import traceback
            traceback.print_exc()
            return

        # Update list holder with the new allometry widget
        self.list_widget_holder.children = (self.allometric_select,)

        # Add observer for dynamic updates when user CHANGES allometry selection
        self.allometric_select.observe(self.update_species_select, names="value")

        # Trigger the species update using the DEFAULT allometry value set above
        print(f"Triggering initial update_species_select call with allometry value: {self.allometric_select.value}") # DEBUG
        self.update_species_select({'new': self.allometric_select.value, 'owner': self.allometric_select, 'type': 'change', 'name': 'value'})
        print("--- End _add_allo_type ---") # DEBUG


    def update_species_select(self, change): # Accept the change dict
        print("--- update_species_select ---") # DEBUG
        print(f"Change event received: name={change.get('name')}, new={change.get('new')}") # DEBUG

        # Clear previous state in the dedicated output area
        with self.species_output_area:
            clear_output(wait=True) # Use wait=True to reduce flicker

        self.widget_species_select = None # Reset placeholder
        self.species_selection_widgets = {} # Reset placeholder

        if not self.name_column_species_allo:
             print("WARN: name_column_species_allo not set.") #DEBUG
             return

        # Ensure allometric_select widget exists and has a value
        if not hasattr(self, 'allometric_select') or not getattr(self.allometric_select, 'value', None):
             print("WARN: allometry type selection is empty or widget doesn't exist.") #DEBUG
             return

        print(f"Updating species list for allometry selection: {self.allometric_select.value}") # DEBUG

        # Get the updated species selection dictionary
        species_selection_dict = self.setup_species_widgets(
            self.name_column_species_allo
        )
        print(f"Result from setup_species_widgets: Keys={list(species_selection_dict.keys()) if species_selection_dict else 'None'}") # DEBUG

        if not species_selection_dict or not species_selection_dict.get("checkboxes"):
             print("WARN: setup_species_widgets did not return valid structure. No species area displayed.") # DEBUG
             # Output area is already cleared above.
             print("--- End update_species_select (failed setup) ---") # DEBUG
             return

        # --- Create the container VBox for checkboxes and dynamic zone widgets ---
        # This VBox will be displayed inside self.species_output_area
        checkbox_vbox = widgets.VBox(list(species_selection_dict["checkboxes"].values()))
        self.widget_species_select = widgets.VBox([checkbox_vbox]) # Holds checkboxes + active zone widgets
        print("Created new species widget area (checkbox container).") # DEBUG

        # --- Store the widgets for each zone ---
        # These VBoxes (species list + output) will be added/removed from self.widget_species_select by the checkbox handler
        self.species_selection_widgets = {
            zone: widgets.VBox(
                [
                    species_selection_dict["species_select_widgets"][zone],
                    species_selection_dict["output_widgets"][zone],
                ], layout=widgets.Layout(margin='0 0 0 20px') # Indent zone widgets
            )
            for zone in species_selection_dict["species_select_widgets"].keys()
        }
        print(f"Stored zone widgets for: {list(self.species_selection_widgets.keys())}") #DEBUG

        # --- Display the Checkbox container in the dedicated Output area ---
        with self.species_output_area:
            # Output area was already cleared with wait=True
            print("Displaying species selection checkboxes...") # Optional status message
            display(self.widget_species_select) # Display the VBox with checkboxes

        print(f"Displayed species widget area in the Output widget.") # DEBUG
        print("--- End update_species_select ---") # DEBUG

    # setup_species_widgets: Keep the debug prints suggested previously inside this method.
    # Ensure it returns {} or an empty dict if filtering fails or combine_list is empty.
    # Make sure the on_checkbox_change handler correctly adds/removes children
    # from self.widget_species_select (the VBox container)
    def setup_species_widgets(self, name_column_species_allo):
        # (Keep the version with detailed debug prints from the previous answer)
        # ... (ensure it handles empty filters gracefully and returns {} on failure) ...

        # --- Make sure the checkbox handler logic is correct ---
        def on_checkbox_change(change):
            checkbox_widget = change['owner']
            zone = ""
             # Find zone key associated with this checkbox instance
            for z, cb in checkboxes.items(): # Use checkboxes dict created in this scope
                 if cb == checkbox_widget:
                      zone = z
                      break
            if not zone:
                 print("WARN: Could not determine zone from checkbox.")
                 return

            target_zone_widgets = self.species_selection_widgets.get(zone)
            if not target_zone_widgets:
                 print(f"WARN: Could not find widgets for zone '{zone}' in self.species_selection_widgets")
                 return

            is_checked = change['new']
            print(f"Checkbox '{checkbox_widget.description}' changed to {is_checked} for zone '{zone}'") #DEBUG

            # Ensure self.widget_species_select (the VBox container) exists
            if not self.widget_species_select:
                 print("ERROR: self.widget_species_select is None in on_checkbox_change!")
                 return

            current_children = list(self.widget_species_select.children)

            if is_checked:
                # Checkbox checked - ADD the zone's widgets VBox if not already present
                if target_zone_widgets not in current_children:
                    print(f"Adding widgets for zone '{zone}'") #DEBUG
                    self.widget_species_select.children = tuple(current_children + [target_zone_widgets])
                else:
                     print(f"Widgets for zone '{zone}' already present.") #DEBUG
                # Trigger update of the output area within the zone widget
                # filter_function(zone, species_selection_wid=species_select_widgets[zone].value) # Pass the *value*
            else:
                # Checkbox unchecked - REMOVE the zone's widgets VBox if present
                if target_zone_widgets in current_children:
                    print(f"Removing widgets for zone '{zone}'") #DEBUG
                    self.widget_species_select.children = tuple(
                        child for child in current_children if child != target_zone_widgets
                    )
                     # Also clear the selection and output explicitly if needed
                    species_select_widgets[zone].value = []
                    with output_widgets[zone]:
                          clear_output()
                else:
                    print(f"Widgets for zone '{zone}' not currently displayed.") #DEBUG

        # ... (rest of setup_species_widgets, returning the dict) ...
        # Make sure you are assigning the on_checkbox_change handler correctly
        for zone, checkbox in checkboxes.items():
            checkbox.observe(on_checkbox_change, names="value")

        # ... return the dict ...

    @property
    def selected_data(self):
        """Returns a dictionary of selected DataFrames for each zone."""
        return {zone: df for zone, df in self.df_tree_selected.items() if not df.empty}

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
