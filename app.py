import streamlit as st
import os
import json
import glob
import random
import pandas as pd
import csv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from supabase import create_client, Client
from streamlit_option_menu import option_menu
from annotated_text import annotated_text
from dotenv import load_dotenv
from datetime import datetime

# Import configuration and utilities
from config import (
    AVAILABLE_MODELS_GROQ,
    AVAILABLE_MODELS_OPENROUTER,
    EXAMPLE_TEXTS,
    PROMPTS
)

# Set page configuration
st.set_page_config(
    page_title="Value Lens",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

# Function to load value theory files
@st.cache_data
def load_value_theories(path="value_theories"):
    value_theories = {}
    for file_path in glob.glob(f"{path}/*.json"):
        try:
            file_name = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                value_theories[file_name[:-5]] = json.load(f)  # Remove .json extension
        except Exception as e:
            st.error(f"Error loading value theory {file_path}: {e}")
    return value_theories

# Function to load datasets from CSV files
@st.cache_data
def load_datasets(path="datasets"):
    datasets = {}
    for file_path in glob.glob(f"{path}/*.csv"):
        try:
            dataset_name = None
            dataset_description = None
            entries = []
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Leer la primera línea para obtener el nombre del dataset
                dataset_name = "🗃️ " + f.readline().strip()
                
                # Leer la segunda línea para obtener la descripción del dataset
                dataset_description = f.readline().strip()
                
                # Leer la tercera línea para obtener los encabezados
                headers_line = f.readline().strip()
                headers = headers_line.split("###") if headers_line else []
                
                # Encontrar los índices de las columnas importantes
                text_idx = -1
                values_idx = -1
                intensity_idx = -1
                model_idx = -1
                
                for i, header in enumerate(headers):
                    header_lower = header.lower()
                    if header_lower == "text":
                        text_idx = i
                    elif header_lower == "values":
                        values_idx = i
                    elif header_lower == "intensity":
                        intensity_idx = i
                    elif header_lower == "model":
                        model_idx = i
                
                # Verificar que se encontraron las columnas necesarias
                if text_idx == -1 or values_idx == -1 or intensity_idx == -1:
                    st.warning(f"Dataset {file_path} no contiene todas las columnas requeridas (text, values, intensity)")
                    continue
                
                # Leer el resto del archivo como entradas CSV
                for line in f:
                    if line.strip():  # Saltar líneas vacías
                        parts = line.strip().split("###")
                        
                        if len(parts) > max(text_idx, values_idx, intensity_idx):
                            # Extraer texto
                            text = parts[text_idx]
                            
                            # Procesar valores
                            values_str = parts[values_idx].strip('[]')
                            values = [v.strip().strip("'\"") for v in values_str.split(',')] if values_str else []
                            values = [v for v in values if v]  # Eliminar valores vacíos
                            
                            # Procesar intensidades
                            intensity_str = parts[intensity_idx].strip('[]')
                            intensities = [i.strip().strip("'\"") for i in intensity_str.split(',')] if intensity_str else []
                            intensities = [i for i in intensities if i]  # Eliminar intensidades vacías
                            
                            entry = {
                                "text": text,
                                "values": values,
                                "intensities": intensities
                            }
                            
                            # Añadir modelo si está disponible
                            if model_idx != -1 and model_idx < len(parts):
                                entry["model"] = parts[model_idx]
                                
                            entries.append(entry)
            
            if dataset_name and entries:
                datasets[dataset_name] = {
                    "description": dataset_description,
                    "entries": entries
                }
        except Exception as e:
            st.error(f"Error loading dataset {file_path}: {e}")
    
    return datasets


@st.cache_resource
def init_llm(model, temperature, api_key, provider="groq"):
    if provider.lower() == "groq":
        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(
            model=model,
            temperature=temperature,
            model_kwargs={"seed": 42}
        )
    else:  # OpenRouter
        os.environ["OPENROUTER_API_KEY"] = api_key
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://valuelens.app"}  # Optional, for tracking            
        )

@st.cache_resource
def get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        st.error("Supabase credentials not set in .env")
        st.stop()
    return create_client(supabase_url, supabase_key)
    
def intensity_horizontal_bar(intensity):
    # Mapeo posicional de intensidad a posiciones
    intensity_positions = {
        "strong_resistance": 0,
        "mild_resistance": 1,
        "neutral": 2,
        "mild_support": 3,
        "strong_support": 4,
        "reframing": 2,
        "no_values": 2
    }
    
    # Dibujar una barra con cursor
    position = intensity_positions.get(intensity, 2)
    bar = ["▫️"] * 5
    bar[position] = "🔘"
    intensity_bar = "".join(bar)
    
    # Mapeo cromático de intensidad a posiciones
    intensity_colors = {
        "strong_resistance": "🟥🟥🟥⬜⬜",
        "mild_resistance": "⬜🟥🟥⬜⬜",
        "neutral": "⬜⬜⬜⬜⬜",
        "mild_support": "⬜⬜🟩🟩⬜",
        "strong_support": "⬜⬜🟩🟩🟩",
        "reframing": "⬜🟨🟨🟨⬜",
        "no_values": "⬛⬛⬛⬛⬛"
    }    

    # Obtener la barra de color correspondiente a la intensidad
    intensity_bar = intensity_colors.get(intensity, "⬜⬜⬜⬜⬜")

    return intensity_bar

# Definimos los colores según intensidad
intensity_colors = {
    "strong_support": "#aa3",       # Verde oscuro
    "mild_support": "#be9",         # Verde claro
    "neutral": "#fea",              # Amarillo
    "mild_resistance": "#faa",      # Rosa claro
    "strong_resistance": "#f88",    # Rojo
    "reframing": "#aaf",            # Azul claro
    "multiple_values": "#eff"       # Gris azulado para múltiples
}

# Función para obtener color según intensidad
def get_color_for_intensity(intensity):
    return intensity_colors.get(intensity, "#eff")

# Función para generar fragmentos anotados de un texto con evidencias
def generate_annotated_fragments(text, evidences):
    # Encontrar todas las posiciones donde hay evidencias
    positions = []
    for evidence in evidences:
        evidence_text = evidence["text"]
        start_pos = 0
        while True:
            pos = text.find(evidence_text, start_pos)
            if pos == -1:
                break
            positions.append({
                "start": pos,
                "end": pos + len(evidence_text),
                "evidence": evidence
            })
            start_pos = pos + 1

    # Ordenar y fusionar posiciones superpuestas
    positions.sort(key=lambda x: x["start"])
    merged_positions = []
    for pos in positions:
        if not merged_positions or pos["start"] > merged_positions[-1]["end"]:
            merged_positions.append({
                "start": pos["start"],
                "end": pos["end"],
                "evidences": [pos["evidence"]]
            })
        else:
            if pos["end"] > merged_positions[-1]["end"]:
                merged_positions[-1]["end"] = pos["end"]
            merged_positions[-1]["evidences"].append(pos["evidence"])

    # Construir fragmentos
    fragments = []
    last_end = 0
    for pos in merged_positions:
        if pos["start"] > last_end:
            fragments.append(text[last_end:pos["start"]])
        segment_text = text[pos["start"]:pos["end"]]
        if len(pos["evidences"]) == 1:
            e = pos["evidences"][0]
            color = get_color_for_intensity(e["intensity"])
            fragments.append((segment_text, e["value_name"], color))
        else:
            value_names = " | ".join(sorted(set(e["value_name"] for e in pos["evidences"])))
            fragments.append((segment_text, value_names, "#e6f7ff"))
        last_end = pos["end"]
    if last_end < len(text):
        fragments.append(text[last_end:])
    return fragments

# Función para recopilar todas las evidencias
def collect_all_evidences(values):
    evidences = []
    for value in values:
        for evidence_text in value.get("evidence", []):
            evidences.append({
                "text": evidence_text,
                "value_name": value.get("name"),
                "value_id": value.get("id"),
                "intensity": value.get("intensity", "neutral")
            })
    return evidences

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

# Initialize session state for persistent variables
if 'value_theories' not in st.session_state:
    st.session_state.value_theories = load_value_theories()

if 'datasets' not in st.session_state:
    st.session_state.datasets = load_datasets()

if 'selected_dataset' not in st.session_state:
    if st.session_state.datasets:
        # Set the first dataset as default
        st.session_state.selected_dataset = list(st.session_state.datasets.keys())[0]
    else:
        st.session_state.selected_dataset = None

if 'detected_values' not in st.session_state:
    st.session_state.detected_values = []

if 'scaled_values' not in st.session_state:
    st.session_state.scaled_values = []

if 'merged_values' not in st.session_state:
    st.session_state.merged_values = []

if 'text_to_analyze' not in st.session_state:
    st.session_state.text_to_analyze = ""    

if 'selected_theory' not in st.session_state:
    st.session_state.selected_theory = None  # Valor inicial    

# ─────────────────────────────────────────────
# AGENT FUNCTION
# ─────────────────────────────────────────────

def run_value_filtering(llm, text_to_analyze, value_theory):
    prompt = PromptTemplate.from_template(PROMPTS["value_filtering"])
    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({
        "value_theory": json.dumps(value_theory, indent=2),
        "text": text_to_analyze
    })


def run_value_measuring(llm, text_to_analyze, value_theory, detected_values):
    prompt = PromptTemplate.from_template(PROMPTS["value_measuring"])
    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({
        "value_theory": json.dumps(value_theory, indent=2),
        "text": text_to_analyze,
        "detected_values": json.dumps(detected_values, indent=2)
    })

def display_analysis_results():

    # Mostrar el subheader
    st.subheader("🔍 Analysis Results", divider="gray")

    # Recopilar todas las evidencias
    all_evidences = collect_all_evidences(st.session_state.merged_values)

    # Mostrar texto con anotaciones
    if not all_evidences:
        st.write(st.session_state.text_to_analyze)
    else:
        fragments = generate_annotated_fragments(st.session_state.text_to_analyze, all_evidences)
        annotated_text(*fragments)

    # Crear una leyenda de colores (más pequeña)
    legend_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;font-size:0.8rem'>"
    for intensity, color in intensity_colors.items():
        legend_html += f"<div style='display:flex;align-items:center;'><div style='width:12px;height:12px;background-color:{color};margin-right:3px;border:1px solid #ccc;'></div><span>{intensity.replace('_', ' ').title()}</span></div>"
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # Muestra el menu de pestañas (tabs)
    result_tabs = st.tabs(["Summary", "Detailed Analysis", "Comparision", "JSON Data"])

    # Tab de Summary
    with result_tabs[0]: 
        summary_data = []
        for value in st.session_state.merged_values:
            intensity = value.get("intensity", "N/A")

            emoji_map = {
                "strong_support": "🔥",
                "mild_support": "👍",
                "neutral": "⚖️",
                "mild_resistance": "🤔",
                "strong_resistance": "🛑",
                "reframing": "🔄",
            }
            emoji = emoji_map.get(intensity, "⚪")

            subjectivity_level = int(value.get("subjectivity_level", 0))
            subjectivity_bar = "◾" * subjectivity_level + "◽" * (4 - subjectivity_level)
            intensity_bar = intensity_horizontal_bar(intensity)

            summary_data.append({
                "Value": f"🆔 {value.get('name')} ({value.get('id')})",
                "Intensity": f"{emoji} {intensity.replace('_', ' ').title()}",
                "Intensity bar": intensity_bar,
                "Subjectivity": subjectivity_bar,
            })
        st.dataframe(summary_data)

    # Tab de Detailed Analysis
    with result_tabs[1]:  # Detailed Analysis
        for value in st.session_state.merged_values:
            intensity = value.get("intensity", "N/A")
            subjectivity_level = int(value.get("subjectivity_level", 0))
            subjectivity_bar = "◾" * subjectivity_level + "◽" * (4 - subjectivity_level)
            intensity_bar = intensity_horizontal_bar(intensity)

            emoji = {
                "strong_support": "🔥",
                "mild_support": "👍",
                "neutral": "⚖️",
                "mild_resistance": "🤔",
                "strong_resistance": "🛑",
                "reframing": "🔄",
            }.get(intensity, "⚪")

            with st.expander(f"🆔 **{value.get('name')}** ({value.get('id')})  ·  {intensity_bar}  ·  {emoji} {intensity.replace('_', ' ').title()}  ·  Subjectivity: {subjectivity_bar}"):
                # Mostrar el texto completo con esta evidencia resaltada usando annotated_text
                #if "evidence" in value and value["evidence"]:
                #    st.write("**📝 Evidence:**")
                #    for evidence in value["evidence"]:
                #        st.badge(evidence, icon=":material/format_quote:", color="orange")
                # Recopilar todas las evidencias para este valor específico
                value_evidence = [{
                    "text": e,
                    "value_name": value.get("name"),
                    "value_id": value.get("id"),
                    "intensity": value.get("intensity", "neutral")
                } for e in value.get("evidence", [])]
                
                if value_evidence:
                    fragments = generate_annotated_fragments(st.session_state.text_to_analyze, value_evidence)
                    annotated_text(*fragments)
                else:
                    st.write(st.session_state.text_to_analyze)
                
                if "justification" in value:
                    #st.write("**🧾 Justification:**")
                    st.markdown(f"<div style='background-color:#FFF3E0;padding:10px;border-radius:10px;margin-bottom:10px;font-size:0.9rem;color:#E65100'>👩🏽‍⚖️️ <b>Justification:</b> {value['justification']}</div>", unsafe_allow_html=True)

    # Nueva pestaña de Comparación
    with result_tabs[2]:  # Comparison
        # Obtener valores detectados por el modelo
        model_values = [value.get("name") for value in st.session_state.merged_values]
        
        # Obtener valores seleccionados por el usuario
        user_values = list(st.session_state.selected_values)
        
        # Convertir IDs de usuario a nombres si es necesario
        theory_values = st.session_state.value_theories.get(st.session_state.selected_theory, {}).get("values", [])
        id_to_name = {value["id"]: value["name"] for value in theory_values}
        user_value_names = [id_to_name.get(value_id, value_id) for value_id in user_values]
        
        # Crear conjuntos para comparación
        model_set = set(model_values)
        user_set = set(user_value_names)
        
        # Valores coincidentes
        matches = model_set.intersection(user_set)
        # Valores solo detectados por el modelo
        model_only = model_set - user_set
        # Valores solo seleccionados por el usuario
        user_only = user_set - model_set
        
        # Crear tabla de comparación
        comparison_data = []
        
        # Añadir coincidencias
        for value in matches:
            comparison_data.append({
                "Value": f"🆔 {value}",
                "User Selection": "✅",
                "Model Detection": "✅",
                "Status": "👏🏽 Coincidencia"
            })
        
        # Añadir valores solo del modelo
        for value in model_only:
            comparison_data.append({
                "Value": f"🆔 {value}",
                "User Selection": "❌",
                "Model Detection": "✅",
                "Status": "🤖 Solo modelo"
            })
        
        # Añadir valores solo del usuario
        for value in user_only:
            comparison_data.append({
                "Value": f"🆔 {value}",
                "User Selection": "✅",
                "Model Detection": "❌",
                "Status": "🫡 Solo usuario"
            })
        
        # Mostrar estadísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coincidencias", len(matches))
        with col2:
            st.metric("Solo modelo", len(model_only))
        with col3:
            st.metric("Solo usuario", len(user_only))
        
        # Mostrar tabla de comparación
        if comparison_data:
            st.dataframe(comparison_data)
        else:
            st.info("No hay valores para comparar.")

    # Tab de JSON result
    with result_tabs[3]: 
        st.code(json.dumps(st.session_state.merged_values, indent=2), language="json", wrap_lines=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

# Use streamlit-option-menu for navigation
with st.sidebar:
    #st.title("Value Lens ⚖️")
    st.image("./images/valuelens-logo.png", use_container_width=True)
    st.caption("An AI-powered tool to explore and analyze text through the lens of human values.") 
    tabs = option_menu(
        None, 
        ["Text Analysis", "Value Theories", "Datasets", "History", "Paper", "Settings"],
        icons=["bar-chart", "book", "database", "clock-history", "file-text", "gear"], 
        default_index=0
    )

# ─────────────────────────────────────────────
# 🔍 TAB 1: TEXT ANALYSIS
# ─────────────────────────────────────────────

# Text Analysis Tab
if tabs == "Text Analysis":
    st.title("🔎 Text Value Analysis")
    st.caption("Select a value theory, enter your text, optionally view the predicted labels, and evaluate how values are expressed within the text.") 
    
    # Prepare theory options with metadata display
    theory_options = []
    theory_display_names = {}
    for theory_name, theory_data in st.session_state.value_theories.items():
        display_name = theory_name
        if "metadata" in theory_data and "theory" in theory_data["metadata"]:
            display_name = "📚 " + theory_data["metadata"]["theory"]
            theory_display_names[theory_name] = display_name
        theory_options.append(theory_name)

    # Select value theory for analysis (using the metadata display name)
    selected_theory = st.selectbox(
        "Select Value Theory",
        options=theory_options,
        index=theory_options.index(st.session_state.selected_theory) if st.session_state.selected_theory in theory_options else 0,
        format_func=lambda x: theory_display_names.get(x, x),
        key="theory_selector"
    )    

    # Text input area
    text_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get("text_to_analyze", ""),
        key="text_input_area",
        height=120
    )

    # Guardar los valores seleccionados en session_state
    st.session_state.selected_theory = selected_theory    
    st.session_state.text_to_analyze = text_input

    # Mostrar los valores disponibles como pills para selección del usuario

    # Inicializar selected_values en session_state si no existe
    if "selected_values" not in st.session_state:
        st.session_state.selected_values = set()

    # Obtener los valores de la teoría seleccionada
    theory_values = st.session_state.value_theories.get(selected_theory, {}).get("values", [])

    # Añadir los dos nuevos tags al principio
    new_tags = [ {"id": "no_values", "name": "No Values"} ]
    theory_values = new_tags + theory_values

    # Crear un mapeo de nombres a IDs para facilitar la conversión
    name_to_id = {value["name"]: value["id"] for value in theory_values}
    id_to_name = {value["id"]: value["name"] for value in theory_values}

    # Obtener los nombres de los valores actualmente seleccionados
    selected_names = [id_to_name[value_id] for value_id in st.session_state.selected_values if value_id in id_to_name]

    # Función de callback para manejar cambios en los pills
    def update_selected_values():
        # Actualizar selected_values basado en las pills seleccionadas
        st.session_state.selected_values = {
            name_to_id[name] for name in st.session_state.value_pills if name in name_to_id
        }

    # Crear pills para cada valor con callback
    selected_pills = st.pills(
        label="Select values detected in the text *(optional)*:",
        options=[value["name"] for value in theory_values],
        default=selected_names,
        selection_mode="multi",
        key="value_pills",
        #disabled=True,
        on_change=update_selected_values
    )
    
    # Dataset selector - Add this new section above the random/clear buttons
    dataset_options = list(st.session_state.datasets.keys()) if st.session_state.datasets else []
    if dataset_options:
        # Obtener la descripción del dataset seleccionado para mostrarla como ayuda
        current_description = ""
        if st.session_state.selected_dataset and st.session_state.selected_dataset in st.session_state.datasets:
            current_description = st.session_state.datasets[st.session_state.selected_dataset].get("description", "")
        
        selected_dataset = st.selectbox(
            "Select Dataset for random examples:",
            options=dataset_options,
            index=dataset_options.index(st.session_state.selected_dataset) if st.session_state.selected_dataset in dataset_options else 0,
            help=current_description
        )
        st.session_state.selected_dataset = selected_dataset
    
    # Generate Random Text y Clear All (50% cada uno)
    col1, col2 = st.columns(2)

    # Generate Random Text button 
    with col1:
        if st.button("🎲 Generate Random Text", type="secondary", use_container_width=True):
            # If we have datasets loaded and one is selected, use that
            if st.session_state.datasets and st.session_state.selected_dataset:
                dataset_entries = st.session_state.datasets[st.session_state.selected_dataset]["entries"]
                if dataset_entries:
                    # Pick a random entry from the selected dataset
                    random_entry = random.choice(dataset_entries)
                    st.session_state.text_to_analyze = random_entry["text"]
                    
                    # Also select the corresponding values in the pills
                    value_names = random_entry["values"]
                    
                    # Find the corresponding IDs in the current theory
                    theory_value_names = {value["name"]: value["id"] for value in theory_values}
                    selected_value_ids = set()
                    
                    # For each value in the dataset entry, find matching value in theory
                    for value_name in value_names:
                        value_name = value_name.strip()
                        if value_name in theory_value_names:
                            selected_value_ids.add(theory_value_names[value_name])
                        elif value_name == "":  # Handle empty value case
                            if "no_values" in name_to_id:
                                selected_value_ids.add("no_values")
                    
                    # Update selected values in session state
                    st.session_state.selected_values = selected_value_ids
            else:
                # Fall back to the original example texts
                st.session_state.text_to_analyze = random.choice(EXAMPLE_TEXTS)
            st.rerun()

    # Clear Text and Values button
    with col2:
        if st.button("🗑️ Clear All", help="Clear text and selected values", use_container_width=True):
            st.session_state.text_to_analyze = ""
            st.session_state.selected_values = set()
            st.rerun()

    # Segunda fila: Analyze Text (100% del ancho)
    analyze_button = st.button("📊 Analyze Text", type="primary", use_container_width=True)
    
    # Analysis button (logic)
    if analyze_button:
        text_to_analyze = st.session_state.text_to_analyze

        if not text_to_analyze:
            st.error("Please enter text to analyze.")
        elif not selected_theory:
            st.error("Please select a value theory.")
        else:
            with st.spinner("Analyzing text values..."):
                # Get the correct API key based on provider
                api_provider = st.session_state.get('api_provider', 'Groq')
                
                if api_provider == 'Groq':
                    api_key = st.session_state.get('groq_api_key', os.getenv('GROQ_API_KEY'))
                    if not api_key:
                        st.error("GROQ API key not set. Please configure it in the Settings tab.")
                        st.stop()
                else:  # OpenRouter
                    api_key = st.session_state.get('openrouter_api_key', os.getenv('OPENROUTER_API_KEY'))
                    if not api_key:
                        st.error("OpenRouter API key not set. Please configure it in the Settings tab.")
                        st.stop()

                llm = init_llm(
                    model=st.session_state.get('selected_model', AVAILABLE_MODELS_GROQ[0]),
                    temperature=st.session_state.get('temperature', 0.0),
                    api_key=api_key,
                    provider=api_provider.lower()
                )
                value_theory = st.session_state.value_theories[selected_theory]

                try:
                    # Step 1: Value Detection
                    detected_values = run_value_filtering(llm, text_to_analyze, value_theory)
                    st.session_state.detected_values = detected_values

                    # Step 2: Intensity Measuring
                    if detected_values:
                        scaled_values = run_value_measuring(llm, text_to_analyze, value_theory, detected_values)
                        st.session_state.scaled_values = scaled_values

                        merged_values = []
                        for detected in detected_values:
                            scaled = next((s for s in scaled_values if s.get("id") == detected.get("id")), {})
                            merged_values.append({**detected, **scaled})
                        st.session_state.merged_values = merged_values
                    else:
                        st.session_state.scaled_values = []
                        st.session_state.merged_values = []

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
            
            # Step 3: Display Results
            if st.session_state.merged_values:
                display_analysis_results()
            elif st.session_state.detected_values == []:
                st.info("No values were detected in the provided text.")
                
            # Guardar en Supabase
            try:
                supabase = get_supabase_client()
                # Get default model based on provider
                default_model = AVAILABLE_MODELS_GROQ[0] if api_provider == 'Groq' else AVAILABLE_MODELS_OPENROUTER[0]
                supabase.table("valuelens").insert({
                    "value_theory": selected_theory,
                    "text": text_to_analyze,
                    "result": json.dumps(merged_values),
                    "llm": st.session_state.get('selected_model', default_model)
                }).execute()
            except Exception as e:
                st.warning(f"⚠️ Could not save to history: {e}")

# ─────────────────────────────────────────────
# 📘 TAB 2: VALUE THEORIES
# ─────────────────────────────────────────────

# Value Theories Tab
elif tabs == "Value Theories":
    st.title("📚 Value Theories")
    st.caption("Browse all loaded value theories with descriptions to understand their principles, assumptions, and use cases.")

    # Select a value theory
    theory_names = list(st.session_state.value_theories.keys())
    
    # Prepare theory options with metadata display
    theory_display_names = {}
    for theory_name, theory_data in st.session_state.value_theories.items():
        display_name = theory_name
        if "metadata" in theory_data and "theory" in theory_data["metadata"]:
            display_name = "📚 " + theory_data["metadata"]["theory"]
            theory_display_names[theory_name] = display_name
    
    selected_theory = st.selectbox(
        "Select a Value Theory model to view its details.",
        options=theory_names,
        index=1 if st.session_state.value_theories else None,
        format_func=lambda x: theory_display_names.get(x, x)
    )
    
    # Display the selected theory
    if selected_theory:
        #st.header(f"Theory: {selected_theory}")
        
        # Mostrar el DataFrame como una tabla
        theory_content = st.session_state.value_theories[selected_theory]
        
        # Obtener el nombre de la teoría desde metadata
        theory_display_name = selected_theory
        if "metadata" in theory_content and "theory" in theory_content["metadata"]:
            theory_display_name = theory_content["metadata"]["theory"]
        
        st.subheader(f"Metadata for `{theory_display_name}`", divider="gray")
        st.dataframe(pd.DataFrame(theory_content["metadata"].items(), columns=["Parameter", "Description"]))
        st.subheader("Values", divider="gray")
        st.session_state.current_theory = st.session_state.value_theories[selected_theory]
        st.dataframe(pd.DataFrame(theory_content["values"]))

        # Format and display the theory content
        st.subheader("JSON", divider="gray")
        st.json(theory_content)
        
        # Formatear el JSON para mostrarlo de forma amigable
        #formatted_json = json.dumps(theory_content, indent=4, ensure_ascii=False)
        #st.code(formatted_json, language="json", wrap_lines=True)
        
    else:
        st.warning("No value theories found. Please add JSON theory files to the 'theories' directory.")

# ─────────────────────────────────────────────
# 📊 TAB 3: DATASETS
# ─────────────────────────────────────────────

# Datasets Tab
elif tabs == "Datasets":
    st.title("📊 Datasets")
    st.caption("Explore available datasets with their descriptions and sample entries for text analysis and value theory research.")

    # Check if datasets are loaded
    if not st.session_state.datasets:
        st.warning("No datasets found. Please add CSV dataset files to the 'datasets' directory.")
    else:
        # Create a selectbox for dataset selection
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset_name = st.selectbox(
            "Select a Dataset to explore:",
            options=dataset_names,
            index=0 if dataset_names else None,
            help="Choose a dataset to view its details and sample entries"
        )

        if selected_dataset_name:
            # Get the selected dataset
            selected_dataset = st.session_state.datasets[selected_dataset_name]

            # Display dataset title and description
            st.subheader(f"{selected_dataset_name}", divider="gray")

            # Show description
            description = selected_dataset.get("description", "No description available")

            # Show sample entries (first 10)
            entries = selected_dataset.get("entries", [])

            if entries:
                # Show total count
                #st.markdown(f"**Description:** {description}")
                st.markdown(f"<div style='background-color: #FFF3E0; padding: 10px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #FF9800;'><strong>📌 Description:</strong> {description}</div>", unsafe_allow_html=True)
                
                st.info(f"🧮 **Total Entries Available:** {len(entries)}")

                # Determine how many entries to show (max 50)
                entries_to_show = min(50, len(entries))

                # Create table data
                table_data = []

                for i, entry in enumerate(entries[:entries_to_show], 1):
                    # Truncate text for better table display (first 100 characters)
                    text = entry.get('text', 'No text available')
                    truncated_text = text[:100] + '...' if len(text) > 100 else text

                    # Format values as comma-separated string
                    values = entry.get('values', [])
                    values_str = ', '.join([f"{v}" for v in values]) if values else 'None'

                    # Format intensities as comma-separated string
                    intensities = entry.get('intensities', [])
                    intensities_str = ', '.join([f"{i}" for i in intensities]) if intensities else 'None'

                    table_data.append({
                        '#': i,
                        'Text': truncated_text,
                        'Values': values_str,
                        'Intensity': intensities_str
                    })

                # Display as dataframe/table
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            '#': st.column_config.NumberColumn(
                                '#',
                                width="small",
                                format="%d"
                            ),
                            'Text': st.column_config.TextColumn(
                                'Text',
                                width="large"
                            ),
                            'Values': st.column_config.TextColumn(
                                'Values',
                                width="medium"
                            ),
                            'Intensity': st.column_config.TextColumn(
                                'Intensity',
                                width="medium"
                            )
                        }
                    )

                    # Show note about truncated text
                    st.caption("💡 **Sample Entries (First 50 are shown)**. *Text column shows first 100 characters.*")
                else:
                    st.warning("No entries found in this dataset.")

# ─────────────────────────────────────────────
# 📄 TAB 4: PAPER
# ─────────────────────────────────────────────
# Paper Tab
elif tabs == "Paper":
    st.title("📄 Value Lens: Using Large Language Models to Understand Human Values")
    
    # Read and display the valuelens.md file
    try:
        with open("valuelens.md", "r", encoding="utf-8") as f:
            paper_content = f.read()
        
        # Display the markdown content
        st.markdown(paper_content)
    except FileNotFoundError:
        st.error("The file 'valuelens.md' was not found in the application directory.")
    except Exception as e:
        st.error(f"Error reading the paper file: {e}")

# ─────────────────────────────────────────────
# ⚙️ TAB 5: SETTINGS
# ─────────────────────────────────────────────
# Configuration Tab
elif tabs == "Settings":
    st.title("⚙️ Settings")
    st.caption("Configure the API provider, choose the LLM model, manage API keys, and tune generation settings like temperature for precise control.")
    
    # API Provider selection
    api_provider = st.selectbox(
        "API Provider",
        options=["Groq", "OpenRouter"],
        index=0 if st.session_state.get('api_provider', 'Groq') == 'Groq' else 1,
        help="Select the API provider for language models"
    )
    st.session_state.api_provider = api_provider
    
    # Model selection based on provider
    if api_provider == "Groq":
        # Check if current model is valid for Groq, otherwise use default
        current_model = st.session_state.get('selected_model', AVAILABLE_MODELS_GROQ[0])
        if current_model not in AVAILABLE_MODELS_GROQ:
            current_model = AVAILABLE_MODELS_GROQ[0]
        
        selected_model = st.selectbox(
            "Select LLM Model",
            options=AVAILABLE_MODELS_GROQ,
            index=AVAILABLE_MODELS_GROQ.index(current_model),
            help="Select Groq language model to analyze the text"
        )
        st.session_state.selected_model = selected_model
        
        # Groq API Key input
        api_key = st.text_input(
            "GROQ API Key",
            value=st.session_state.get('groq_api_key', os.getenv('GROQ_API_KEY', '')),
            type="password",
            help="Enter your GROQ API key"
        )
        st.session_state.groq_api_key = api_key
    else:  # OpenRouter
        # Check if current model is valid for OpenRouter, otherwise use default
        current_model = st.session_state.get('selected_model', AVAILABLE_MODELS_OPENROUTER[0])
        if current_model not in AVAILABLE_MODELS_OPENROUTER:
            current_model = AVAILABLE_MODELS_OPENROUTER[0]
        
        selected_model = st.selectbox(
            "Select LLM Model",
            options=AVAILABLE_MODELS_OPENROUTER,
            index=AVAILABLE_MODELS_OPENROUTER.index(current_model),
            help="Select OpenRouter language model to analyze the text"
        )
        st.session_state.selected_model = selected_model
        
        # OpenRouter API Key input
        api_key = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.get('openrouter_api_key', os.getenv('OPENROUTER_API_KEY', '')),
            type="password",
            help="Enter your OpenRouter API key"
        )
        st.session_state.openrouter_api_key = api_key
    
    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get('temperature', 0.0),
        step=0.1,
        help="Lower values make the output more deterministic, higher values more creative"
    )
    st.session_state.temperature = temperature
    
    # Save configuration
    #if st.button("Save Configuration", use_container_width=True):
    #    st.success("Configuration saved successfully!")
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 📘 TAB 5: History
# ─────────────────────────────────────────────
# History Tab

elif tabs == "History":
    st.title("🕓 Analysis History")
    st.caption("View recent analyses with timestamps, value theories, detected values, and model details. Expand any entry to inspect annotated text and raw JSON results.")      
    
    try:
        supabase = get_supabase_client()
        response = supabase.table("valuelens").select("*").order("created_at", desc=True).limit(10).execute()
        records = response.data
        if records:
            for record in records:
                # Convertir el string ISO a objeto datetime
                formatted_datetime = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00')).strftime("%d %b %Y - %H:%M")

                # Muestra los badges oportunos
                values = json.loads(record["result"])

                # Crear contenedor de badges
                badges_html = ""

                # 0. Badge de fecha y hora (formateada), con fondo blanco y borde azul oscuro
                badges_html += f'<span style="background-color: #fff; color: #0D47A1; border: 1px solid #0D47A1; padding: 2px 7px; border-radius: 5px; margin-right: 5px; font-size: 0.8em; display: inline-block;">📚 {formatted_datetime}</span>'

                # 1. Badge del value_theory (al inicio)
                badges_html += f'<span style="background-color: #1E88E5; color: #fff; padding: 2px 7px; border-radius: 5px; margin-right: 5px; font-size: 0.8em; display: inline-block;">📚 {record["value_theory"]}</span>'

                # 2. Badges de los valores detectados
                for v in values:
                    color = intensity_colors.get(v['intensity'], "#ccc")  # Color según intensidad
                    badges_html += f'<span style="background-color: {color}; color: #222; padding: 2px 7px; border-radius: 5px; margin-right: 5px; font-size: 0.8em; display: inline-block;">⚖️ {v["name"]} &nbsp;</span>'

                # 3. Badge del modelo de lenguaje (al final)
                badges_html += f'<span style="background-color: #888; color: #fff; padding: 2px 7px; border-radius: 5px; margin-right: 5px; font-size: 0.8em; display: inline-block;">🤖 {record["llm"]}</span>'

                # Mostrar todo junto
                st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;">{badges_html}</div>', unsafe_allow_html=True)

                with st.expander("🔍 Ver detalles"):
                #with st.expander(f"📆 {record['value_theory']} · {formatted_datetime}"):

                    # Mostrar texto con anotaciones
                    try:
                        values = json.loads(record["result"])
                        # Recopilar todas las evidencias
                        all_evidences = []
                        for value in values:
                            for evidence_text in value.get("evidence", []):
                                all_evidences.append({
                                    "text": evidence_text,
                                    "value_name": value.get("name"),
                                    "value_id": value.get("id"),
                                    "intensity": value.get("intensity", "neutral")
                                })
                        
                        # Generar fragmentos anotados
                        if all_evidences:
                            fragments = generate_annotated_fragments(record['text'], all_evidences)
                            annotated_text(*fragments)
                        else:
                            st.write(record['text'])
                            
                        # Crear una leyenda de colores (más pequeña)
                        legend_html = "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;font-size:0.8rem'>"
                        for intensity, color in intensity_colors.items():
                            legend_html += f"<div style='display:flex;align-items:center;'><div style='width:12px;height:12px;background-color:{color};margin-right:3px;border:1px solid #ccc;'></div><span>{intensity.replace('_', ' ').title()}</span></div>"
                        legend_html += "</div>"
                        st.markdown(legend_html, unsafe_allow_html=True)

                    except Exception as e:
                        st.warning(f"Could not parse result JSON: {e}")
                        st.markdown(f"<div style='background-color:#fff3e0;padding:10px;border-radius:10px;margin-bottom:10px;font-size:0.9rem;color:#e65100'>{record['text']}</div>", unsafe_allow_html=True)
                    
                    st.json(json.loads(record["result"]), expanded=False)
        else:
            st.info("No history records found.")
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")


# Footer
st.sidebar.markdown("---")
st.sidebar.image("./images/urjc-logo.png", use_container_width=True, caption="Madrid, Spain")
st.sidebar.image("./images/qr-code-github.png", use_container_width=True)
#st.sidebar.caption("© 2025 Value Lens Analysis Demo App")
