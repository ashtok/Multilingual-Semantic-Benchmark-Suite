data = {
    "Language": ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Japanese",
                 "Korean",
                 "Arabic", "Turkish", "Dutch", "Polish", "Swedish", "Norwegian (Bokmål)", "Danish", "Finnish", "Czech",
                 "Romanian",
                 "Hungarian", "Ukrainian", "Hebrew", "Bulgarian", "Greek", "Croatian", "Serbian", "Slovak", "Slovenian",
                 "Lithuanian",
                 "Latvian", "Estonian", "Thai", "Vietnamese", "Malay", "Persian", "Indonesian", "Tamil", "Hindi",
                 "Bengali",
                 "Swahili", "Icelandic", "Maltese", "Irish", "Welsh", "Bosnian", "Georgian", "Amharic", "Uzbek",
                 "Tagalog"],
    "Synsets": [14123602, 8067721, 8339254, 7405208, 5974637, 5197205, 5122269, 4140873, 4247016, 3054437,
                3775816, 3189556, 8835657, 4794877, 6140113, 3167194, 4158363, 4681364, 3641483, 3777925,
                3568611, 3949107, 2949208, 3288792, 2811488, 3005293, 3042220, 3084388, 4710301, 2862522,
                2735807, 3071481, 2650559, 3553123, 3104299, 3350950, 3514483, 2603136, 2756319, 3094122,
                2865133, 2860448, 2636057, 4837838, 3292605, 2569269, 2574305, 2445371, 2608407, 2741530],
    "Senses": [33697253, 13253333, 13740456, 13020304, 9371558, 7763993, 11265693, 7645768, 7488820, 4588807,
               6515291, 4254121, 12835025, 7334440, 12626951, 3717798, 4980096, 6187313, 4938388, 5143646,
               4585048, 6560846, 3864902, 4051955, 3429036, 3495884, 6124234, 3627977, 5234525, 3396466,
               3229645, 3701625, 3120784, 5386787, 3812110, 6695981, 5153085, 2819523, 3175344, 3500096,
               3118442, 3140724, 2779626, 5218774, 3706922, 2782735, 2828024, 2476397, 3167906, 3134109]
}


def count_and_sort_languages():
    """Count languages and sort by synsets in descending order"""

    # Total number of languages
    total_languages = len(data["Language"])
    print(f"Total number of languages: {total_languages}")
    print("\nLanguages ranked by Synsets (descending order):")
    print("=" * 30)

    # Create list of tuples (language, synsets) and sort by synsets descending
    language_data = list(zip(data["Language"], data["Synsets"]))
    sorted_data = sorted(language_data, key=lambda x: x[1], reverse=True)

    # Print sorted results - rank and name only
    for i, (language, synsets) in enumerate(sorted_data, 1):
        print(f"{i:2d}. {language}")

    return sorted_data


# Run the function
if __name__ == "__main__":
    sorted_languages = count_and_sort_languages()