from babelnet import Language

LANGUAGE_CONFIG = {
    'high_resource': {
        Language.EN: {'name': 'English', 'code': 'en'},
        Language.ES: {'name': 'Spanish', 'code': 'es'},
        Language.FR: {'name': 'French', 'code': 'fr'},
        Language.DE: {'name': 'German', 'code': 'de'},
        Language.IT: {'name': 'Italian', 'code': 'it'},
        Language.PT: {'name': 'Portuguese', 'code': 'pt'},
        Language.RU: {'name': 'Russian', 'code': 'ru'},
        Language.ZH: {'name': 'Chinese', 'code': 'zh'},
        Language.JA: {'name': 'Japanese', 'code': 'ja'},
        Language.KO: {'name': 'Korean', 'code': 'ko'},
        Language.AR: {'name': 'Arabic', 'code': 'ar'},
        Language.TR: {'name': 'Turkish', 'code': 'tr'},
        Language.NL: {'name': 'Dutch', 'code': 'nl'},
        Language.PL: {'name': 'Polish', 'code': 'pl'},
        Language.SV: {'name': 'Swedish', 'code': 'sv'},
        Language.NO: {'name': 'Norwegian', 'code': 'no'},
        Language.DA: {'name': 'Danish', 'code': 'da'},
        Language.FI: {'name': 'Finnish', 'code': 'fi'},
        Language.CS: {'name': 'Czech', 'code': 'cs'},
        Language.RO: {'name': 'Romanian', 'code': 'ro'},
        Language.HU: {'name': 'Hungarian', 'code': 'hu'},
        Language.UK: {'name': 'Ukrainian', 'code': 'uk'},
        Language.HE: {'name': 'Hebrew', 'code': 'he'},
        Language.BG: {'name': 'Bulgarian', 'code': 'bg'},
        Language.EL: {'name': 'Greek', 'code': 'el'}
    },

    'medium_resource': {
        Language.HR: {'name': 'Croatian', 'code': 'hr'},
        Language.SR: {'name': 'Serbian', 'code': 'sr'},
        Language.SK: {'name': 'Slovak', 'code': 'sk'},
        Language.SL: {'name': 'Slovenian', 'code': 'sl'},
        Language.LT: {'name': 'Lithuanian', 'code': 'lt'},
        Language.LV: {'name': 'Latvian', 'code': 'lv'},
        Language.ET: {'name': 'Estonian', 'code': 'et'},
        Language.TH: {'name': 'Thai', 'code': 'th'},
        Language.VI: {'name': 'Vietnamese', 'code': 'vi'},
        Language.MS: {'name': 'Malay', 'code': 'ms'},
        Language.FA: {'name': 'Persian', 'code': 'fa'},
        Language.ID: {'name': 'Indonesian', 'code': 'id'},
        Language.TA: {'name': 'Tamil', 'code': 'ta'},
        Language.HI: {'name': 'Hindi', 'code': 'hi'},
        Language.BN: {'name': 'Bengali', 'code': 'bn'}
    },

    'low_resource': {
        Language.SW: {'name': 'Swahili', 'code': 'sw'},
        Language.IS: {'name': 'Icelandic', 'code': 'is'},
        Language.MT: {'name': 'Maltese', 'code': 'mt'},
        Language.GA: {'name': 'Irish', 'code': 'ga'},
        Language.CY: {'name': 'Welsh', 'code': 'cy'},
        Language.BS: {'name': 'Bosnian', 'code': 'bs'},
        Language.KA: {'name': 'Georgian', 'code': 'ka'},
        Language.AM: {'name': 'Amharic', 'code': 'am'},
        Language.UZ: {'name': 'Uzbek', 'code': 'uz'},
        Language.TL: {'name': 'Tagalog', 'code': 'tl'}
    }
}
