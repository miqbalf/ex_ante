# function to automate filter selection
def filter_or_selection(df_string_var, column_name, *args):

    # we will use eval, allometric_column_filter (df) will be hard-coded or as string

    # filter_df_string = 'allometric_column_filter['
    # for i in range(len(args)):
    #     if args[-1] != args[i]:
    #         filter_df_string += f'(allometric_column_filter["{column_name}"] == "{args[i]}" ) |'
    #     else:
    #         filter_df_string += f'(allometric_column_filter["{column_name}"] == "{args[i]}" )]'

    filter_df_string = f"{df_string_var}["
    for i in range(len(args)):
        if args[-1] != args[i]:
            filter_df_string += f'({df_string_var}["{column_name}"] == "{args[i]}" ) |'
        else:
            filter_df_string += f'({df_string_var}["{column_name}"] == "{args[i]}" )]'

    # allometric_country_selected = eval(filter_df_string)

    # return allometric_country_selected

    return filter_df_string
