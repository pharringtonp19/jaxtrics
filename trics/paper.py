def generate_latex_table(data, label, caption, note=None):
    headers = ['Model', 'Est', 'Std', '\%\Delta', 'N', 'Params', 'Core', 'Tenant', 'Landlord']
    
    # Begin table with threeparttable for notes
    latex_table = f'\\begin{{table}}[htbp]\n\\begin{{threeparttable}}\n\\centering\n\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n'
    latex_table += '\\begin{tabular}{@{}l c c c c c c c c c@{}}\n'
    
    # Double horizontal line
    latex_table += '\\toprule\n\\toprule\n'

    # Headers
    latex_table += ' & '.join(headers) + '\\\\ \\midrule\n'
    
    # Rows
    for item in data:
        row = []
        for key in headers:
            value = item[key]
            if isinstance(value, bool):
                value = '\\checkmark' if value else ''
            row.append(str(value))
        latex_table += ' & '.join(row) + '\\\\\n'
    
    # Horizontal line at the end
    latex_table += '\\bottomrule\n'

    # End of tabular
    latex_table += '\\end{tabular}\n'
    
    if note:
        # Add the note at the end of the table
        latex_table += '\\begin{tablenotes}\n\\small\n\\item ' + note + '\\end{tablenotes}\n'

    # End of threeparttable and table
    latex_table += '\\end{threeparttable}\n\\end{table}'
    
    return latex_table