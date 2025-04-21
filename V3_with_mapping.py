import pandas as pd

medical_mapping = {
    'original_firstorder_10Percentile': 'Hypo-dense region intensity (minimum)',
    'original_firstorder_90Percentile': 'Hyper-dense region intensity (maximum)',
    'original_firstorder_Energy': 'Total signal energy in the tumor',
    'original_firstorder_Entropy': 'Tissue heterogeneity (entropy)',
    'original_firstorder_InterquartileRange': 'Interquartile intensity range',
    'original_firstorder_Kurtosis': 'Tissue uniformity (kurtosis)',
    'original_firstorder_Mean': 'Average tissue density',
    'original_firstorder_Median': 'Median tissue density',
    'original_firstorder_Variance': 'Density variance',
    'original_glcm_Contrast': 'Edge sharpness (contrast)',
    'original_glcm_Correlation': 'Tissue texture correlation', 
    'original_glszm_SmallAreaEmphasis': 'Small uniform tissue regions',
    'original_ngtdm_Busyness': 'Tumor texture complexity',
    'Label': 'Tumor classification'
}

# Function to generate medical text based on user inputs
# This function prompts the user to input values for various medical parameters and stores them in a dictionary.
def generate_medical_text_with_input():
    columns = [
        'Hypo-dense region intensity (minimum)',
        'Hyper-dense region intensity (maximum)',
        'Total signal energy in the tumor',
        'Tissue heterogeneity (entropy)',
        'Interquartile intensity range',
        'Tissue uniformity (kurtosis)',
        'Average tissue density',
        'Median tissue density',
        'Density variance',
        'Edge sharpness (contrast)',
        'Tissue texture correlation',
        'Small uniform tissue regions',
        'Tumor texture complexity',
        'Tumor classification'
    ]

    case = {}
    print("Please enter the values for each parameter. Leave blank to skip and use default values.")

    for column in columns:
        user_input = input(f"Enter value for {column}: ")
        case[column] = float(user_input) if user_input.strip() and user_input.replace('.', '', 1).isdigit() else "N/A"

    return case

# Function to interpret a single parameter value
# This function takes a parameter value and its name, then provides an interpretation of the value based on predefined rules.
def interpret_value(value, parameter_name):
    if value == "N/A":
        return f"No value provided for {parameter_name}."
    
    if parameter_name == 'Hypo-dense region intensity (minimum)':
        if value < 20:
            return "This low intensity value suggests the presence of necrotic or cystic tissue, which could be indicative of areas with reduced metabolic activity."
        elif value >= 20 and value <= 50:
            return "A moderate intensity value suggests a mixture of healthy and necrotic tissue, indicating possible tumor heterogeneity."
        else:
            return "This higher intensity suggests a region of dense fibrous or calcified tissue, potentially indicating areas of structural stability within the tumor."
    
    if parameter_name == 'Hyper-dense region intensity (maximum)':
        if value < 30:
            return "Low intensity in this region suggests soft tissue, which could indicate a predominance of non-calcified components."
        elif value >= 30 and value <= 70:
            return "This suggests a mix of soft tissue with possible calcification, commonly seen in tumors with fibrous components."
        else:
            return "High intensity indicates significant calcification or fibrous tissue, which may be indicative of a more stable or organized region of the tumor."

    if parameter_name == 'Total signal energy in the tumor':
        if value < 50:
            return "Low energy suggests less metabolic activity in the tumor, which may indicate a less aggressive or less active region."
        elif value >= 50 and value <= 150:
            return "Moderate energy suggests normal metabolic activity, typical of tumors with varying degrees of aggressiveness."
        else:
            return "High signal energy reflects increased metabolic activity, often associated with more active or aggressive regions of the tumor."
    
    if parameter_name == 'Tissue heterogeneity (entropy)':
        if value < 1:
            return "A low entropy value suggests uniform tissue composition, potentially indicating a more organized tumor structure."
        elif value >= 1 and value <= 3:
            return "Moderate entropy indicates some degree of tissue heterogeneity, which could be found in tumors with mixed cell types."
        else:
            return "High entropy suggests significant heterogeneity in the tumor tissue, a feature often associated with irregular tumor structures."
    
    if parameter_name == 'Interquartile intensity range':
        if value < 10:
            return "A narrow intensity range suggests relatively uniform tissue, which is commonly seen in tumors with less structural variation."
        elif value >= 10 and value <= 30:
            return "A moderate intensity range suggests a mix of more and less dense regions, pointing to some heterogeneity in the tumor."
        else:
            return "A wide intensity range indicates marked heterogeneity in the tumor, with areas of both high and low density."
    
    if parameter_name == 'Tissue uniformity (kurtosis)':
        if value < 2:
            return "A low kurtosis value indicates uneven tissue distribution, often observed in tumors with mixed cell populations."
        elif value >= 2 and value <= 5:
            return "Moderate kurtosis suggests a mix of uniform and irregular areas in the tumor, commonly seen in both benign and malignant tumors."
        else:
            return "A high kurtosis value suggests more uniform tissue, often observed in tumors that are less irregular in structure."
    
    if parameter_name == 'Average tissue density':
        if value < 30:
            return "Low average tissue density suggests a soft or potentially cystic component within the tumor."
        elif value >= 30 and value <= 60:
            return "Moderate tissue density, indicating a mixture of soft and firm tissue structures."
        else:
            return "High average tissue density suggests a solid or potentially fibrous component, often indicative of a more rigid tumor structure."
    
    if parameter_name == 'Median tissue density':
        if value < 30:
            return "A low median density indicates a tumor with a predominance of less dense tissue."
        elif value >= 30 and value <= 60:
            return "A moderate median density reflects a balanced mix of soft and firmer tissue types."
        else:
            return "A high median density suggests more solid, fibrous or calcified tissue regions, typically indicative of a more stable tumor structure."
    
    if parameter_name == 'Density variance':
        if value < 10:
            return "Low density variance indicates a more homogenous tissue structure, with less variation in tissue composition."
        elif value >= 10 and value <= 30:
            return "Moderate density variance suggests some heterogeneity within the tumor, with areas of different tissue densities."
        else:
            return "High density variance indicates significant variation in tissue types across the tumor, a feature often associated with irregular or aggressive tumors."
    
    if parameter_name == 'Edge sharpness (contrast)':
        if value < 1:
            return "Poor edge sharpness suggests that the tumor margins are poorly defined, which could be indicative of invasive or irregular tumor growth."
        elif value >= 1 and value <= 3:
            return "Moderate edge sharpness indicates moderately defined margins, suggesting that the tumor may have a less invasive growth pattern."
        else:
            return "High edge sharpness suggests well-defined tumor margins, typically seen in more organized or less aggressive tumors."
    
    if parameter_name == 'Tissue texture correlation':
        if value < 0.5:
            return "Low correlation suggests that the tumor has a disorganized structure, often seen in aggressive or irregular tumors."
        elif value >= 0.5 and value <= 0.7:
            return "Moderate correlation suggests some organization of tissue structure, with a mix of regular and irregular patterns."
        else:
            return "High correlation indicates well-organized tissue structure, typical of less aggressive tumors."
    
    if parameter_name == 'Small uniform tissue regions':
        if value < 5:
            return "Few small uniform tissue regions suggest that the tumor is more complex and potentially more aggressive."
        elif value >= 5 and value <= 20:
            return "Moderate number of small uniform tissue regions, indicating a mix of organized and disorganized tissue."
        else:
            return "A high number of small uniform tissue regions suggests a more organized structure, typically seen in benign or less aggressive tumors."
    
    if parameter_name == 'Tumor texture complexity':
        if value < 2:
            return "Low texture complexity suggests that the tumor has a simple structure, often observed in benign tumors."
        elif value >= 2 and value <= 4:
            return "Moderate texture complexity suggests that the tumor has some irregularity, commonly seen in tumors with mixed characteristics."
        else:
            return "High texture complexity indicates a highly irregular structure, often characteristic of malignant tumors."

    return "No interpretation available for this parameter."

# Function to generate a detailed analysis of all parameters
# This function loops through all the parameters in the case dictionary and generates a detailed analysis text for each.
def generate_full_text(case):
    text = "Detailed Medical Parameter Analysis:\n\n"
    for key, value in case.items():
        text += f"{key}: {value}\n"
        text += interpret_value(value, key) + "\n\n"
    return text

# Function to generate the final report
# This function compiles the full parameter analysis and adds general insights to create the final radiological report.
def generate_report(case):
    report = "Radiological Evaluation Report:\n\n"
    report += generate_full_text(case)

    report += "The following insights have been gathered from the imaging data:\n"
    report += "- Tumor tissue heterogeneity has been identified, suggesting areas of varied structure and metabolic activity, which may require careful assessment during treatment planning.\n"
    report += "- The tumor shows regions of dense and soft tissue, with varying degrees of metabolic activity. Surgical planning should consider these areas for targeted resection.\n"
    report += "- Regular monitoring of the tumor’s growth and structure may be necessary to adapt treatment strategies as the tumor evolves.\n"
    report += "These results should be correlated with clinical data to formulate an optimal management plan tailored to the patient.\n"
    
    return report

# Main execution block
# This is the main part of the program that executes the above functions to generate and save the medical report.
if __name__ == "__main__":
    case = generate_medical_text_with_input()
    medical_text = generate_report(case)

    print("\nGenerated Medical Report:\n")
    print(medical_text)

    with open('D:/Bioinformatics dept/Project/details/dataafterfilter/generated_medical_text.txt', 'w') as file:
        file.write(medical_text)

    print("\nMedical text has been saved to 'D:/Bioinformatics dept/Project/details/dataafterfilter/generated_medical_text.txt'.")


