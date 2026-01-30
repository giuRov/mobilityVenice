from typing import Any, Dict, Tuple
import pandas as pd
from .utils import normalise_title

def add_ticket_class(
    validation_data: pd.DataFrame,
    *,
    title_col: str = "title_description",
    out_col: str = "ticket_class",
    user_category_col: str = "user_category",
) -> pd.DataFrame:
    """
    Assign ticket_class based on title_description (exact lists with light normalisation),
    and add user_category derived from ticket_class.
    """

    df = validation_data.copy()

    t_75min = {
        "BIGL RETE UNICA 75'", "BIGL.AUT.75'MESTRE/LIDO-TSC",
        "75'-TPL 8,64-COMVE0,86", "75'-TPL 7,47-COMVE2,03",
        "75'-TPL 6,64-COMVE0,86", "75'-TPL 6,30-COMVE1,20",
        "PEOPLEMOVER+BUS+TRAM 75'", "75'-TPL 13,28-COMVE1,72",
        "75'-TPL 12,60-COMVE2,40", "BIGL.MESTRE/LIDO 75' A BORDO",
        "ORD. NAVIGAZIONE 75' ONLINE", "BIGLIETTO DI BORDO CV 75'",
        "BORDO 75MIN CARTAVENEZIA", "75'-TPL 12,60-CVE2,40 ONLINE",
        "75'-TPL 6,00-COMVE1,50", "PEOPLEMOVER+BUS+TRAM 75'CARNET",
        "NA-BIG.AUT.75' MESTRE/LIDO-CSC", "ORD. NAVIG. 75' ONLINE 1 MESE",
        "NA-CARNET NAV. 10 CORSE DA 75'", "VENDITA A BORDO 75' ORD.",
        "VENDITA A BORDO 75' CV", "NA-C AUT. 10 CORSE 75' CARD",
        "NA75'-TPL 13,28-COMVE1,72", "C AUT.10 CORSE 75' TICKET NA",
    }

    t_24h = {
        "DAILYP-TPL19,90-C.VE5,10", "24H-TPL 14,90-COM.VE5,10",
        "DAILYP-TPL19,71-C.VE5,29", "DAILY PASS VENEZIA - AVM",
        "DAILY PASS VENEZIA ONLINE", "CAV -TREP + ACTV 24H",
        "JESOLO + ACTV 24H", "24ORE ONLINE NO AEROBUS",
        "24H METROPOLITANO ORD+1", "CATR+AVM24H-TPL24,71-C.VE5,29",
        "NA-24H METROPOLITANO ORD+2", "NA-24H METROPOLITANO ORD+1",
        "T.FUSINA VE+ACTV 24 ORE", "24HAERCS-TPL26,90-CVE5,10",
        "DAILY PASS VE. ONLINE 1MESE", "JESO+AVM24H-TPL25,91-C.VE5,29",
        "24HAERCS-TPL20,9-CVE5,1", "24H METROPOLITANO ORD+2",
        "24HAERCS-TPL25,21-CVE6,79", "24HAERCS-TPL22,90-CVE5,10",
        "FUSIVE+AVM24H-TPL26,51-CVE4,49", "24ORE ONLINE AEROBUS CS",
        "24HAERAR-TPL29,92-CVE8,08", "24HAERAR-TPL32,90-CVE5,10",
        "24HAERAR-TPL26,9-CVE5,1", "CAORLE-P.S.MARGH. + ACTV 24H",
        "24ORE ONLINE AEROBUS AR", "24HAERAR-TPL28,90-CVE5,10",
        "24H ONLINE AEROBUS CS 1 MESE", "BIBIONE + ACTV 24H",
        "24H-24 ORE", "24H METROPOLITANO ORD ONLINE",
        "NA-24H-TPL 14,90-COM.VE5,10", "24H METROPOLITANO ORD+2 ONLINE",
        "CASM+AVM24H-TPL28,71-C.VE5,29", "BIBI+AVM24H-TPL31,91-C.VE5,29",
        "24H METROPOLITANO ORD+1 ONLINE", "24H METROPOLITANO ORD.",
        "VILLE VENETE+24H ACTV URB+NAV", "24H ONLINE AEROBUS AR 1 MESE",
        "ERACLEAMARE + ACTV 24H", "VILLE V+24H TPL28,26 C.VE 5,34",
        "VILLE V+24H TPL28,50 C.VE 5,10", "NA-24H METROPOLITANO ORD.",
        "ERMA+AVM24H-TPL26,91-C.VE5,29", "ARRIVA MISTO ACTV 24H",
        "NA-24H METROPOLITANO RES+1", "24H METROPOLITANO RES+1",
        "LIGNANO + ACTV 24H", "24H METROPOLITANO RES.",
        "LIGN+AVM24H-TPL31,91-C.VE5,29", "NA-24H METROPOLITANO RES.",
        "24H METROPOLITANO RES+2", "NA-24H METROPOLITANO RES+2",
        "VILLE VENETE SOLO LINEA 53 24H", "CAMP.MARINA+ACTV 24H",
        "VILLE VENETE L.53 24H ONLINE", "24HMESTRE-TPL4,04-C.VE0,96",
        "24HVR-VE-TPL38,36-C.VE5,14",
    }

    t_48h = {
        "48H-TPL 24,90-COMVE5,10", "48H-TPL 29,90-COMVE5,10",
        "48H-TPL 27,50-COMVE7,50", "48ORE ONLINE NO AEROBUS",
        "48H ONLINE NO AEROBUS 1MESE", "48HAERCS-TPL36,90-CVE5,10",
        "48HAERCS-TPL30,9-CVE5,1", "48ORE ONLINE AEROBUS CS",
        "48HAERCS-TPL33,00-CVE9,00", "48HAERCS-TPL31,90-CVE5,10",
        "48ORE ONLINE AEROBUS AR", "48HAERAR-TPL42,90-CVE5,10",
        "48HAERAR-TPL36,9-CVE5,1", "48HAERAR-TPL37,90-CVE5,10",
        "48HAERAR-TPL37,72-CVE10,28", "48H ONLINE AEROBUS CS 1 MESE",
        "48H ONLINE AEROBUS AR 1 MESE",
    }

    t_72h = {
        "72H-TPL 33,40-COMVE6,60", "72H-TPL 38,40-COMVE6,60",
        "BIGLIETTO 72 ORE ROLL. VENICE", "72ORE ONLINE NO AEROBUS",
        "72H-TPL 35,36-COMVE9,64", "72 ORE R.VENICE ONLINE",
        "72H ROLL.VEN-TPL21,22-C.VE5,78", "72 ORE R.VENICE+AEROPORTO CS",
        "72H ONLINE NO AEROBUS 1MESE", "72ORE ONLINE AEROBUS AR",
        "72H R.VENICE ONLINE 1 MESE", "72 ORE R.VENICE+AEROPORTO AR",
        "72ORE ONLINE AEROBUS CS", "72H RVENICE+AEROP.CS ONLINE",
        "72HAERAR-TPL51,40-CVE6,60", "72H R.VENICE+AEROP.AR ONLINE",
        "72HAERAR-TPL45,4-CVE6,60", "72HAERCS-TPL45,40-CVE6,60",
        "72HAERCS-TPL39,4-CVE6,60", "ATVO CANOVA+ACTV 72H ONLINE",
        "72HAERCS-TPL40,86-CVE11,14", "72HAERCS-TPL40,40-CVE6,60",
        "72HAERAR-TPL45,58-CVE12,42", "72HAERAR-TPL46,40-CVE6,60",
        "ATVOCANOVA+ACTV 72HROLL.ONLINE", "T.FUSINA VE+ACTV 72 ORE",
        "72HRVE+AERCS-TPL26,72-C.VE7,28", "72H RVE+AEROP.CS ONLINE 1 MESE",
        "72H ONLINE AEROBUS CS 1 MESE", "72HRVE+AERAR-TPL31,43-C.VE8,57",
        "72H ONLINE AEROBUS AR 1 MESE", "ATVO CANOVA+ACTV 72H",
        "CAV - TREP + ACTV 72H", "72H R.VE.+AER.AR ONLINE 1MESE",
        "FUSIVE+AVM72H-TPL54,01-CVE8,99", "CATR+AVM72H-TPL50,36-C.VE9,64",
        "ATVO CANOVA+ACTV 72H ROLLING",
    }

    t_7days = {
        "7GG-TPL 43,60-COMVE16,40", "7 DAYS ONLINE NO AEROBUS",
        "7GG-TPL 48,60-COMVE16,40", "7GG-TPL 51,08-COMVE13,92",
        "7 DAYS ONLINE AEROBUS AR", "7 DAYS ONLINE NO AEROBUS 1MESE",
        "7GGAERAR-TPL55,6-CVE16,4", "7GGAERAR-TPL61,60-CVE16,40",
        "7 DAYS ONLINE AEROBUS CS", "7GGAERAR-TPL61,29-CVE16,71",
        "7GGAERAR-TPL56,60-CVE16,40", "7 DAYS ONLINE AEROBUS AR 1MESE",
        "7GGAERCS-TPL49,6-CVE16,4", "7GGAERCS-TPL55,60-CVE16,40",
        "7GGAERCS-TPL56,58-CVE15,42", "7GGAERCS-TPL50,60-CVE16,40",
        "7 DAYS ONLINE AEROBUS CS 1MESE",
    }

    t_stud_month = {
        "MENS.STUDENTE RETE UNICA", "MENSILE STUDENTE ISOLE", "MENSILE STUDENTE EXTRA",
        "ATVO+ACTV MENS.STUD.F1", "ATVO+ACTV MENS.STUD.F2", "ATVO+ACTV MENS.STUD.F3",
        "MENSILE STUD. PELLESTRINA", "ABB. STUDENTE MENS. CHIOGGIA", "MENS. STUDENTE BUS LIDO",
    }

    t_stud_year = {
        "ABB STUD. RETEUNICA 12 MESI", "ANNUALE STUDENTE ISOLE", "ANNUALE STUDENTE EXTRA",
        "STUD. RETE INTERA  FAMILIARE", "ABB.STUD.ANN.PELLESTRINA",
        "ATVO+ACTV ANN.STUD.F1", "ATVO+ACTV ANN.STUD.F2", "ATVO+ACTV ANN.STUD.F3",
        "SUPP. 12 MESI STUDENTE LAGUNA", "ABB STUD. 12 MESI CHIOGGIA",
        "S.TERRR+ACTV STUDENTE TR.6", "S.TERR+ACTV ANN STUD TR.6",
        "S.TERR+ACTV STUDENTE TR.2", "S.TERR+ACTV ANN STUD TR.2",
        "MOBILITY STUDENTE RETEUNICA", "STUDENTE EXTRA FAMILIARE", "S.TERR+ACTV ANN STUD TR.1",
        "S.TERR+ACTV STUDENTE TR.3", "S.TERR+ACTV STUDENTE TR.7",
        "S.TERR+ACTV ANN STUD TR.8", "S.TERR+ACTV ANN STUD TR.7",
        "ABB STUDENTE BUS LIDO 12 MESI", "S.TERR+ACTV STUDENTE TR.5",
        "S.TERR+ACTV ANN STUD TR.3", "STUDENTE CHIOGGIA FAMILIARE",
        "S.TERR+ACTV STUDENTE TR.1", "S.TERR+ACTV STUDENTE TR.4",
        "S.TERR+ACTV ANN STUD TR.5", "S.TERR+ACTV ANN STUD TR.9",
        "SUPP. 12 MESI STUDENTE AUTOMOB", "S.TERR+ACTV STUDENTE TR.8",
        "S.TERR+ACTV STUDENTE TR.9", "S.TERR+ACTV ANN STUD TR.4",
    }

    t_ret_year = {
        "ABB. OVER75 GRATUITO", "ABBONAMENTO PENSIONATI ACTV",
        "ABB. OVER75 RETE UNICA 50%", "ABB. OVER 75 A20", "ABB. OVER 75 A5",
    }

    #t_wkrs_month = {"ATVO+ACTV MENS.LAV.F1", "ATVO+ACTV MENS.LAV.F2", "ATVO+ACTV MENS.LAV.F3"}
    #t_wkrs_year = {"ATVO+ACTV ANN.LAV.F1", "ATVO+ACTV ANN.LAV.F2", "ATVO+ACTV ANN.LAV.F3"}

    t_res_month = {
        "MENSILE ORDINARIO RETE UNICA", "MENSILE ORDINARIO ISOLE", "MENSILE ORDINARIO EXTRA",
        "SUPP MENS.NAVIGAZIONE", "MENSILE ORD. RES. PELLESTRINA", "ABB. MENSILE CHIOGGIA",
        "ATVO+ACTV MENS.ORD.F1", "ABBONAMENTO 30 GG.PEOPLEMOVER", "ATVO+ACTV MENS.ORD.F2",
        "ABB MENSILE PEOPLEMOVER", "SUPP MENS.AUTOMOBILISTICO",
        "ATVO+ACTV MENS.20%.F1", "ATVO+ACTV MENS.20%.F2", "ATVO+ACTV MENS.20%.F3",
        "MENS. COSE ANIMALI RETE INTERA", "MENS. COSE ANIMALI RETE UNICA",
        "MENSILE PARK+RETE INTERA", "ATVO+ACTV MENS.ORD.F3",
        "ARRIVA AEROPORTO O.MENS", "DDGR1201-1297/2022 R.UNICA",
        "ATVO+ACTV MENS.5%.F2", "SUPP MENS.URBANO CHIOGGIA",
        "SUPP MENSILE PEOPLEMOVER", "DDGR1201-1297/2022 EXTRA",
        "ATVO+ACTV MENS.LAV.F1", "ATVO+ACTV MENS.LAV.F2", "ATVO+ACTV MENS.LAV.F3"
    }

    t_res_year = {
        "ANNUALE ORDINARIO RETE UNICA", "ANNUALE ORDINARIO ISOLE", "ANNUALE ORD.RES.PELLESTRINA",
        "SUPP.ANNUALE NAVIGAZIONE", "ANNUALE ORDINARIO EXTRA", "ABB ANNUALE PEOPLEMOVER",
        "ANNUALE CAT. D 17(UN SEMESTRE)", "SUPP ANNUALE PEOPLEMOVER", "ABB.CHIOGGIA ANNUALE",
        "SUPP. ANNUALE AUTOMOB.", "S.TERR+ACTV ANN ORD TR.9", "ANNUALE CAT. D LINEA 11",
        "ANNUALE ORDINARIO BUS LIDO", "S.TERR+ACTV ANN ORD TR.2", "S.TERR+ACTV ANN ORD TR.6",
        "S.TERR+ACTV ANN ORD TR.8", "S.TERR+ACTV ANN ORD TR.7", "S.TERR+ACTV ANN ORD TR.4",
        "ANNUALE PARK+RETE INTERA", "S.TERR+ACTV ANN ORD TR.3", "S.TERR+ACTV ANN ORD TR.5",
        "SUPP. 12 MESI CHIOGGIA",
        "ATVO+ACTV ANN.LAV.F1", "ATVO+ACTV ANN.LAV.F2", "ATVO+ACTV ANN.LAV.F3"
    }

    mapping: Dict[Any, str] = {}

    def _add(titles, code: str) -> None:
        for t in titles:
            mapping[normalise_title(t)] = code

    _add(t_24h, "1")
    _add(t_48h, "2")
    _add(t_72h, "3")
    _add(t_7days, "4")
    _add(t_stud_month, "5-STUD")
    _add(t_stud_year, "6-STUD")
    _add(t_ret_year, "6-RET")
    _add(t_res_month, "5-RES")
    _add(t_res_year, "6-RES")
    _add(t_75min, "7")

    norm_series = df[title_col].map(normalise_title)
    df[out_col] = norm_series.map(mapping)

    # user_category (British English spellings already ok)
    user_category_map: Dict[str, str] = {
        "1": "Tourists",
        "2": "Tourists",
        "3": "Tourists",
        "4": "Tourists",
        "5-STUD": "Students",
        "6-STUD": "Students",
        "5-RES": "Residents",
        "6-RES": "Residents",
        "6-RET": "Retirees",
        "7": "Occasional Travellers",
    }
    df[user_category_col] = df[out_col].map(user_category_map)

    return df


def drop_nan_ticket_class(
    df: pd.DataFrame,
    *,
    ticket_col: str = "ticket_class",
    title_col: str = "title_description",
    return_stats: bool = True,
    with_counts: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Drop rows with missing ticket_class and report what was removed.

    Returns:
    - cleaned_df
    - stats dict (if return_stats=True) including:
        - total_before, total_after
        - removed_rows, removed_percentage
        - removed_unique_titles (list)
        - removed_titles_counts (dict) if with_counts=True
    """

    total_before = len(df)
    mask_remove = df[ticket_col].isna()
    removed_rows = int(mask_remove.sum())
    total_after = total_before - removed_rows

    cleaned = df.loc[~mask_remove].copy()

    removed_unique_titles = (
        df.loc[mask_remove, title_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    stats: Dict[str, Any] = {
        "total_before": total_before,
        "total_after": total_after,
        "removed_rows": removed_rows,
        "removed_percentage": round((removed_rows / total_before * 100), 2) if total_before else 0.0,
        "removed_unique_titles_count": len(removed_unique_titles),
        "removed_unique_titles": removed_unique_titles,
    }

    if with_counts:
        removed_counts = (
            df.loc[mask_remove, title_col]
            .dropna()
            .astype(str)
            .value_counts()
        )
        stats["removed_titles_counts"] = removed_counts.to_dict()

    return (cleaned, stats) if return_stats else (cleaned, {})
