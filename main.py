import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_uploader as du
import fitz
from PIL import Image
from io import BytesIO
import base64
import os
import json
from openai import OpenAI 
from anthropic import Anthropic

# Initialize LLMs API clients
open_ai_api_key = os.getenv('open_ai_api_key')
anthropic_api_key = os.getenv('anthropic_api_key')

claude_client = Anthropic(api_key=anthropic_api_key)
open_ai_client = OpenAI(api_key=open_ai_api_key)

port = int(os.environ.get("PORT", 8050))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
du.configure_upload(app, "uploads")

PROMPT_TO_FIND_CONTENT = """You are a helpful assistant assisting a biologist investigating the role of various interventions in obesity and body weight.
You are provided with a text extracted from a PDF page of a scientific publication.

Your task is to determine if the given text contains a plot or table legend description that explicitly compares body weight gain, loss, or change between an intervention group and a control group, targeting a specific protein in an obese mouse model.

The following criteria will help you make the decision:

Key Bodyweight Focus:
The page must explicitly mention body weight, weight gain, weight loss, or weight change as a measured outcome. 
Ignore pages discussing other measurements (e.g., glucose levels, energy expenditure, caloric intake).

Obese Model Requirement:
The text must indicate that the study involves an obese mouse model. 
Terms such as:
HFD (high-fat diet), HD (high diet), HFHC (high fat high carbohydrate), DIO (diet-induced obesity) must be present

Intervention vs. Control:
The intervention and control groups should be mentioned, labeled with terms like KO (gene knockout), -/-, Tg, or specific pharmacological names, in conjunction with the obese model.

Format Indicators:
The text must explicitly describe a table or figure legend, indicated by phrases such as:
"Table 1. Comparison of bodyweight gain..."
"Fig. 3. Bodyweight loss comparison..."

Exclusion Rule:
Exclude pages where the described measurements do not involve body weight. 

Response Rules:
Return only two words:

True: If the text contains a plot or table legend describing a comparison of body weight between intervention and control groups.
False: Otherwise.
Important: Use these rules strictly, and if there is any ambiguity or lack of clear evidence about bodyweight measurements, return "False."

Examples:
Text:
"...Fig. 5.
Increased energy expenditure without any changes in net energy intake in HFD-fed Casp1−/− mice 16 wk. 
(A) Cumulative food intake of HFD-fed animals. 
(B) Fecal output in HFD-fed wild-type and Casp1−/− animals. 
(C) Fat content of feces from HFD-fed wild-type and Casp1−/−animals..."

Output:
False
Reason: Although this is a figure legend description but there is no mention of body weight measurements in the text.

Text:
"Fig. 1.
Absence of the Nlrp3-inﬂammasome protects against the development of HFD-induced obesity. 
(A) qPCR analysis of caspase-1 gene expression levels in
epididymal WAT of LFD- and HFD-fed wild-type C57/
Bl6 animals after 16 wk of diet-intervention. 
(B) Comparison of bodyweight gain in wild-
type, Nlrp3−/−(C), ASC−/−(D), and Casp1−/−(E) animals
on LFD or HFD during 16 wk."

Output:
True
Reason: 
The text of the figure legend explicitly mentions a comparison of body weight gain between intervention and control groups in an obese mouse model.

TEXT:
"""


PROMPT_TO_EXTRACT_CONTENT = """You are a helpful assistant that helps a biologist investigating the role of various proteins in obesity and body weight.
You are provided with an image containing a plot and legends showing the effect of an intervention (targeting a specific protein) on body weight in an obese mouse model.
Your task is to analyze the image and extract data about the effect sizes plotted and return results as a JSON object.

Follow these steps:

**Step 1 – Identify the Figure Type:**
Examine the provided image and determine the figure type:
*   **line:** Time-series plot comparing body weight between groups over time.
*   **bar:** Bar plot comparing body weight between groups at the start and/or end of the experiment.
*   **table:** Table presenting the experimental results.

**Step 2 – Identify Intervention and Control Groups:**
Identify the two sample groups based on the image's text and labels:
*   **Intervention obese group:** This group represents the intervention and may be labeled with terms like: KO, -/-, Tg, or a pharmacological treatment name.
*   **Non-intervention obese group (Control):** This group serves as the control and may be labeled with terms like: WT, Vehicle, +/+, or Placebo.
Contextual information:
*   Obesity models may be labeled as: HFD (high-fat diet), HD (high diet), or DIO (diet-induced obese).
Examples:
*   NLRP3 protein study:
    *   Intervention: NLRP3 -/- HFD (homozygous knockout NLRP3 + high-fat diet)
    *   Control: WT HFD (wild type + high-fat diet)
*   Beloranib medication study:
    *   Intervention: DIO, Beloranib (diet-induced obese + treatment)
    *   Control: DIO, Vehicle (diet-induced obese + vehicle/placebo)
Prioritization rules:
*   If both heterozygous (+/-) and homozygous (-/-) genetic interventions are present, select the homozygous (-/-) group as the intervention group.
*   If multiple doses of a pharmacological intervention are used, select the highest or most frequent dose as the intervention group.
*   If various combinations of pharmacological interventions are used, select the group with the widest combination as the intervention group.

**Step 3 – Extract Body Weight Values:**
For the identified intervention and control groups, extract the body weight values at the start and end of the experiment:
*   **Line graph:** Extract the y-values (body weight) of the leftmost and rightmost data points for each group.
*   **Bar graph:** Use tick labels to identify the bars corresponding to the start and end time points.
*   **Table:** Find the corresponding rows or columns for the start and end of the experiment.
Data handling:
*   Ignore error bars.
*   If only end-point data is available, return 'None' for the start values.
*   If data extraction fails, return 'None'.

**Step 4 – Estimate Experiment Duration:**
Estimate the experiment duration in weeks:
*   **Line graph:** Calculate the amount of the ticks on the time axis.
*   **Bar graph:** Read the tick labels and take the largest time value.
*   **Table:** Look for corresponding text in row or column names.
Data handling:
*   Convert days to weeks (days / 7).
*   If data extraction fails, return 'None'.
*   Numeration of the ticks on time axis might be with gaps, so for correct estimation you need add amount of the ticks after last number in time axis

**Step 5 – Format the Response as a JSON object:**
Return the results in the following format without any additions, only this key: value pairs:

type_of_the_plot: [line, bar, table];
treatment_group: [name of the treatment group];
control_group: [name of the control group];
treatment_group_start_value: [value or None];
control_group_start_value: [value or None];
treatment_group_final_value: [value or None];
control_group_final_value: [value or None];
duration_of_the_experiment: [value in weeks or None]

** Please check once againg that your responce contains only JSON by following examples format without(!!) any additional text**
Example 1:
{
    "type_of_the_plot": "table";
    "treatment_group": "HD KO";
    "control_group": "HD WT";
    "treatment_group_start_value": "None";
    "control_group_start_value": "None";
    "treatment_group_final_value": 14.9;
    "control_group_final_value": 15.9;
    "duration_of_the_experiment": 12
}

Example 2:
{
    "type_of_the_plot": "line";
    "treatment_group": "NLRP3 -/- HFD";
    "control_group": "WT HFD";
    "treatment_group_start_value": 100;
    "control_group_start_value": 100;
    "treatment_group_final_value": 190;
    "control_group_final_value": 240;
    "duration_of_the_experiment": 17
}
"""


app.layout = html.Div([
    dbc.Row([
        html.H1("OCR Tool for Biologists", className="text-center mb-4")
    ]),
    
    dbc.Row([
        # Left Column
        dbc.Col([
            html.Div([
                # Collapsible uploader
                dbc.Collapse(
                    du.Upload(
                        id='upload-pdf', 
                        text='Drag and Drop PDF File or Click to Select', 
                        max_file_size=50
                    ),
                    id='uploader-collapse',
                    is_open=False  # Initially open
                ),
                html.Div(id='output-pdf-upload', style={'display': 'none'}),  # Hidden backend element
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id='page-selector',
                            placeholder='Select a page',
                            className="my-3",
                        ),
                    ], width=3),
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button(# Button to toggle the uploader
                                "Upload PDF",
                                id='toggle-uploader-button',
                                # color="green",
                                className="me-1",
                                style={'color': 'white', 'background-color': 'green'}
                            ),
                            dbc.Button('<< First Page', id='first-page-button', className="me-1"),
                            dbc.Button('< Prev Page', id='prev-page-button', className="me-1"),
                            dbc.Button('Next Page >', id='next-page-button', className="me-1"),
                            dbc.Button('Last Page >>', id='last-page-button'),
                        ], className="mt-3"),
                    ], width=5),
                ]),
                html.Div(
                    id='page-content',
                    style={
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'min-height': '500px',
                        'max-height': '800px',  # Add maximum height
                        'overflow': 'auto',
                        'display': 'flex',
                        'justify-content': 'center',
                        'align-items': 'center'
                    }
                )
            ])
        ], width=8, style={'border-right': '1px solid #ddd', 'padding': '10px'}),
        
        # Right Column
        dbc.Col([
            # Finding Content Section
            dbc.Card([
                dbc.CardHeader("Finding Content", style={'background-color': '#ffcc99'}),
                dbc.CardBody([
                    dbc.Collapse(
                        dcc.Textarea(
                            id='finding-content-prompt',
                            value=PROMPT_TO_FIND_CONTENT,
                            style={'width': '100%', 'height': '100px'}
                        ),
                        id='finding-content-collapse',
                        is_open=False,
                    ),
                    dbc.Button("Toggle Prompt", id='finding-content-toggle', color="secondary", className="mt-2"),
                    dcc.Dropdown(
                        id='finding-content-model',
                        options=[
                            {'label': 'gpt-4o', 'value': 'gpt-4o'},
                            {'label': 'claude-3-5-sonnet-20241022', 'value': 'claude-3-5-sonnet-20241022'}
                        ],
                        value='gpt-4o',
                        placeholder='Select Model',
                        className="mt-2"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button('Run all pages', id='run-all-pages-button', className="mt-2"),
                        ], width=6),
                        dbc.Col([
                            dbc.Button('Run selected page', id='run-selected-page-button', className="mt-2"),
                        ], width=6),
                    ]),
                    dbc.Progress(id='finding-content-progress', value=0, className="mt-2", style={'display': 'none'}),
                    html.Div(id='finding-content-error', className="mt-3", style={'color': 'red'}),
                ])
            ], className="mb-3"),
            
            # Extracting Content Section
            dbc.Card([
                dbc.CardHeader("Extracting Content", style={'background-color': '#d4edda'}),
                dbc.CardBody([
                    dbc.Collapse(
                        dcc.Textarea(
                            id='extracting-content-prompt',
                            value=PROMPT_TO_EXTRACT_CONTENT,
                            style={'width': '100%', 'height': '100px'}
                        ),
                        id='extracting-content-collapse',
                        is_open=False,
                    ),
                    dbc.Button("Toggle Prompt", id='extracting-content-toggle', color="secondary", className="mt-2"),
                    dcc.Dropdown(
                        id='extracting-content-model',
                        options=[
                            {'label': 'gpt-4o', 'value': 'gpt-4o'},
                            {'label': 'claude-3-5-sonnet-20241022', 'value': 'claude-3-5-sonnet-20241022'}
                        ],
                        value='gpt-4o',
                        placeholder='Select Model',
                        className="mt-2"
                    ),
                    dbc.Button('Run Extracting Content', id='run-extracting-content-button', className="mt-2"),
                    html.Div([
                        html.H5("Results", className="mt-3"),
                        html.Div(
                            id='extracting-content-results',
                            style={
                                'border': '1px solid #ddd',
                                'padding': '10px',
                                'max-height': '200px',
                                'overflow': 'auto',
                                'background-color': '#f9f9f9'
                            }
                        ),
                    ]),
                    dbc.Button('Save Results', id='save-results-button', className="mt-2", disabled=True),
                    html.Div(id='extracting-content-error', className="mt-3", style={'color': 'red'}),
                ])
            ])
        ], width=4, style={'padding': '20px'})
    ]),
    
    html.Div(id='hidden-file-path', style={'display': 'none'})
])

# Callback to toggle the uploader collapse
@app.callback(
    Output('uploader-collapse', 'is_open'),
    [Input('toggle-uploader-button', 'n_clicks'),
     Input('upload-pdf', 'isCompleted')],
    [State('uploader-collapse', 'is_open')]
)
def toggle_uploader_collapse(toggle_clicks, is_completed, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'toggle-uploader-button':
        return not is_open  # Toggle the collapse when the button is clicked
    elif trigger_id == 'upload-pdf' and is_completed:
        return False  # Collapse the uploader after upload is completed

    return is_open

# Callbacks for toggling prompts
@app.callback(
    Output('finding-content-collapse', 'is_open'),
    Input('finding-content-toggle', 'n_clicks'),
    State('finding-content-collapse', 'is_open')
)
def toggle_finding_content_prompt(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

@app.callback(
    Output('extracting-content-collapse', 'is_open'),
    Input('extracting-content-toggle', 'n_clicks'),
    State('extracting-content-collapse', 'is_open')
)
def toggle_extracting_content_prompt(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for updating page selector and displaying content
@app.callback(
    [Output('output-pdf-upload', 'children'),
     Output('page-selector', 'options'),
     Output('hidden-file-path', 'children'),
     Output('page-selector', 'value')],
    [Input('upload-pdf', 'isCompleted'),
     Input('first-page-button', 'n_clicks'),
     Input('prev-page-button', 'n_clicks'),
     Input('next-page-button', 'n_clicks'),
     Input('last-page-button', 'n_clicks')],
    [State('page-selector', 'value'),
     State('page-selector', 'options'),
     State('upload-pdf', 'fileNames'),
     State('upload-pdf', 'upload_id')],
    prevent_initial_call=True
)
def update_page_selector_and_output(is_completed, first_clicks, prev_clicks, next_clicks, last_clicks, current_page, options, filenames, upload_id):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "", [], "", None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'upload-pdf' and is_completed and filenames:
        file_path = os.path.join("uploads", upload_id, filenames[0])
        doc = fitz.open(file_path)
        options = [{'label': f'Page {i+1} of {len(doc)}', 'value': i} for i in range(len(doc))]
        return html.Div([html.P(f"Uploaded: {filenames[0]}")]), options, file_path, 0

    total_pages = len(options) if options else 0

    if trigger_id == 'first-page-button':
        return dash.no_update, dash.no_update, dash.no_update, 0
    elif trigger_id == 'prev-page-button':
        return dash.no_update, dash.no_update, dash.no_update, max(0, current_page - 1)
    elif trigger_id == 'next-page-button':
        return dash.no_update, dash.no_update, dash.no_update, min(total_pages - 1, current_page + 1)
    elif trigger_id == 'last-page-button':
        return dash.no_update, dash.no_update, dash.no_update, total_pages - 1

    return dash.no_update, dash.no_update, dash.no_update, current_page

# Callback for displaying page content
@app.callback(
    Output('page-content', 'children'),
    Input('page-selector', 'value'),
    State('hidden-file-path', 'children')
)
def display_page_content(page_num, file_path):
    if page_num is not None and file_path:
        doc = fitz.open(file_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        img_data = f"data:image/png;base64,{img_str}"
        return html.Div([
            html.Img(
                src=img_data,
                style={
                    'max-width': '100%',
                    'max-height': '780px',
                    'height': 'auto',
                    'object-fit': 'contain'
                }
            )
        ])
    return ""

# Callback for running the "Finding content" task on all pages
# Callback for running the "Finding content" task on all pages
@app.callback(
    [Output('page-selector', 'value', allow_duplicate=True),  # Automatically display the first matching page
     Output('finding-content-error', 'children')],  # Display results or errors
    Input('run-all-pages-button', 'n_clicks'),
    [State('hidden-file-path', 'children'),
     State('finding-content-prompt', 'value'),
     State('finding-content-model', 'value'),
     State('page-selector', 'options')],
    prevent_initial_call=True
)
def run_finding_content_all_pages(n_clicks, file_path, prompt, model, options):
    if n_clicks is None or file_path is None:
        raise dash.exceptions.PreventUpdate

    doc = fitz.open(file_path)
    total_pages = len(options)

    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        try:
            result = perform_finding_content(model, prompt, text)
            if result == "True":
                # Return the first matching page number (0-based index) and a success message
                return page_num, html.Span(f"Found matching content on page {page_num + 1}.", style={'color': 'green'})
        except Exception as e:
            # Return an error message if something goes wrong
            return dash.no_update, f"Error processing page {page_num + 1}: {str(e)}"

    # If no matching page is found, return a message
    return dash.no_update, html.Span("No matching pages found.", style={'color': 'red'})


# Callback for running the "Finding content" task on the selected page
@app.callback(
    [Output('finding-content-error', 'children', allow_duplicate=True)],
    Input('run-selected-page-button', 'n_clicks'),
    [State('hidden-file-path', 'children'),
     State('finding-content-prompt', 'value'),
     State('finding-content-model', 'value'),
     State('page-selector', 'value')],
    prevent_initial_call=True
)
def run_finding_content_selected_page(n_clicks, file_path, prompt, model, page_num):
    if n_clicks is None or file_path is None or page_num is None:
        return [""]

    doc = fitz.open(file_path)
    page = doc.load_page(page_num)
    text = page.get_text("text")
    try:
        result = perform_finding_content(model, prompt, text)
        if result == "True":
            return [html.Span(f"Page {page_num + 1} matches the criteria.", style={'color': 'green'})]
        else:
            return [html.Span(f"Page {page_num + 1} does not match the criteria.", style={'color': 'red'})]
    except Exception as e:
        return [f"Error: {str(e)}"]

# Callback for running the "Extracting content" task
@app.callback(
    [Output('extracting-content-results', 'children'),
     Output('save-results-button', 'disabled'),
     Output('extracting-content-error', 'children')],
    Input('run-extracting-content-button', 'n_clicks'),
    [State('hidden-file-path', 'children'),
     State('extracting-content-prompt', 'value'),
     State('extracting-content-model', 'value'),
     State('page-selector', 'value')],
    prevent_initial_call=True
)
def run_extracting_content(n_clicks, file_path, prompt, model, page_num):
    if n_clicks is None or file_path is None or page_num is None:
        return "", True, ""

    doc = fitz.open(file_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    try:
        result = perform_extracting_content(model, prompt, img_str)
        return html.Pre(result), False, ""
    except Exception as e:
        return "", True, f"Error: {str(e)}"

# Function to perform LLM tasks based on the selected model
def perform_finding_content(model, prompt, text):
    if model == 'gpt-4o':
        response = open_ai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': prompt + text
                }
            ],
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
    
    elif model == 'claude-3-5-sonnet-20241022':
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt + text
                        }
                    ],
                }
            ]
        )
        result = message.to_dict()['content'][0]['text'].strip()
    return result
    
def perform_extracting_content(model, prompt, image):
    if model == 'gpt-4o':
        response = open_ai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": prompt
                },
                {
                    "role": "user", "content": [
                        {
                            "type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image}"}
                        }
                    ]
                }
            ],
            temperature=0.0,
        )
        result = response.choices[0].message.content
    elif model == 'claude-3-5-sonnet-20241022':
        media_type = "image/png"
        message = claude_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        result = message.to_dict()['content'][0]['text']
    return result


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port)
