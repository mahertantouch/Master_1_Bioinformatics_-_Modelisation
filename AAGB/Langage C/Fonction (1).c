/*pFG et pFD sont les ss arbres
abr -> pS est la seq ADN et sa structure
abr est le noeud */

//1.1.1 Libération de la mémoire
void libererABR(tyABR *abr) { //la fonction va permettre de libérer toute la mémoire des arbres binaires
    if (abr == NULL) return;
    
    // lzration des ss-arbres gauche et droit
    libererABR(abr->pFG);
    libererABR(abr->pFD);
    
    // pour libzration de la seq d'ADN
    FreeSeq(abr->pS->seq);  // Libérer la séquence
    free(abr->pS);          // Libérer la structure contenant la séquence
    
    // pour libzrer noeud actuel
    free(abr);
}

//Appel de ces fonctions
/* Libération de l'arbre des ORF et des CDS
libererABR(abrORF);
libererABR(abrCDS); */

//1.1.2 Comparaison ORF et CDS
/*Arg = abrORF : pointeur vers arbre des ORF et abrCDS (vers cds) ; Retour : nombre de seq communes entre les deux arbres. La fonction parcourt les arbres et compare chaque seq. Si ageles, la fonction incrémente le compteur. */
int comparerSeq(char *seq1, char *seq2) {
    return strcmp(seq1, seq2) == 0;  // va donner 1 si identiques, 0 si pas identique
}

void comparerORFandCDS(tyABR *abrORF, tyABR *abrCDS) {
    if (abrORF == NULL || abrCDS == NULL) {
        printf("Un des arbres est vide\n");
        return;
    }

    int countCommon = 0;  // compteur de correspondances entre ORF et CDS.
    int totalORF = nb_noeud(abrORF);  // nolbre total de ORF
    int totalCDS = nb_noeud(abrCDS);  // nbre total de CDS
    
    // Parcours des ORF et comparaison avec les CDS
    tyABR *currentORF = abrORF;
    while (currentORF != NULL) {
        tyABR *currentCDS = abrCDS;
        while (currentCDS != NULL) {
            if (comparerSeq(currentORF->pS->seq, currentCDS->pS->seq)) {
                countCommon++;
                break;  // si la correspondance est trouvée, on arrete de chercher pour cet ORF
            }
            currentCDS = currentCDS->pFD;
        }
        currentORF = currentORF->pFD;
    }
    
    // pour l'zffichage des resultats
    printf("Sur les %d ORF trouvées :\n", totalORF);
    printf("%d (%f%%) des ORF sont des CDS\n", countCommon, (100.0 * countCommon) / totalORF);
    printf("%d (%f%%) des ORF ne sont pas des CDS\n", totalORF - countCommon, (100.0 * (totalORF - countCommon)) / totalORF);
    printf("%d CDS (%f%%) n'ont pas été trouvées\n", totalCDS - countCommon, (100.0 * (totalCDS - countCommon)) / totalCDS);
}

//1.1.3 Recherche ORF dans brin complémentaire
/*Arg = pS : pointeur vers une seq ADN ; Retour : pointeur vers arbre binaire des ORF ; Expl : modifié pour effectuer une recherche d'ORF dans le brin compl. Après avoir trouvé les ORF direct, elle génère le complémentaire et effectue la meme recherche */
tyABR* findORF_ABR(tySeqADN *pS) {
    tyABR *pABR = NULL;
    int iSeq;
    int tIStart[3] = {-1, -1, -1};
    
    // poyr la recherche des ORF dans le brin direct
    for (iSeq = 0; iSeq < pS->lg - 2; iSeq++) {
        if (estStart(pS->seq + iSeq)) {
            if (tIStart[iSeq % 3] == -1) {
                tIStart[iSeq % 3] = iSeq;
            }
        } else if (estStop(pS->seq + iSeq)) {
            if (tIStart[iSeq % 3] != -1) {
                pABR = Ajouter_ABR(pABR, NewSeqADN(iSeq - tIStart[iSeq % 3] + 3, pS->seq + tIStart[iSeq % 3]));
                tIStart[iSeq % 3] = -1;
            }
        }
    }

    // pour rechercher les ORF dans le brin complémentaire
    char *seq_complementaire = BrinComplementaire(pS->seq, pS->lg);
    for (iSeq = 0; iSeq < pS->lg - 2; iSeq++) {
        if (estStart(seq_complementaire + iSeq)) {
            if (tIStart[iSeq % 3] == -1) {
                tIStart[iSeq % 3] = iSeq;
            }
        } else if (estStop(seq_complementaire + iSeq)) {
            if (tIStart[iSeq % 3] != -1) {
                pABR = Ajouter_ABR(pABR, NewSeqADN(iSeq - tIStart[iSeq % 3] + 3, seq_complementaire + tIStart[iSeq % 3]));
                tIStart[iSeq % 3] = -1;
            }
        }
    }
    
    free(seq_complementaire);
    return pABR;
}

//1.2.1 Car. des CDS et ORF dans fichiers
void ecrire_caracteristiques(tyABR *abr, const char *nomFichier) {
    FILE *fichier = fopen(nomFichier, "w");
    if (!fichier) {
        fprintf(stderr, "Erreur lors de l'ouverture du fichier.\n");
        return;
    }

    fprintf(fichier, "lg GC GC1 GC2 GC3\n");

    while (abr != NULL) {
        float gc = calculer_gc(abr->pS->seq, abr->pS->lg);
        float gc1 = calculer_gc_phase(abr->pS->seq, abr->pS->lg, 1);
        float gc2 = calculer_gc_phase(abr->pS->seq, abr->pS->lg, 2);
        float gc3 = calculer_gc_phase(abr->pS->seq, abr->pS->lg, 3);

        fprintf(fichier, "%d %.6f %.6f %.6f %.6f\n",
                abr->pS->lg, gc, gc1, gc2, gc3);

        abr = abr->pFD;  // Parcours de l'arbre
    }

    fclose(fichier);
}

/* arg = abr : arbre des seq (orf ou cds) et le nom de fichier de sortie ; Retour : aucun ; Expl : cette fonction parcourt l'arbre binaire des seq et calcul les caractéristiques pour chaque seq qui sont enregistrés ds un  fichier */

//1.2.2 Filtrage des ORF
void filtrerORF(tyABR *abr, int minLg, float minGC3, float maxGC3) {
    if (abr == NULL) return;
    
    // Calcul du GC3
    float GC1, GC2, GC3;
    All_GC(abr->pS->seq, abr->pS->lg, &GC1, &GC2, &GC3);
    
    // Filtrage par longueur et GC3
    if (abr->pS->lg < minLg || GC3 < minGC3 || GC3 > maxGC3) {
        // Marque cette ORF comme filtrée (ici 0 = filtrée, 1 = non filtrée)
        abr->pS->isFiltered = 0;
    } else {
        abr->pS->isFiltered = 1;
    }
    
    // appels de maniere recursive pour filtrer les sous-arbres gauche et droit
    filtrerORF(abr->pFG, minLg, minGC3, maxGC3); //pour filtrer le ss-arbre gauche
    filtrerORF(abr->pFD, minLg, minGC3, maxGC3);  //pour filtrer le ss-arbre droit
}

/*arg = abr : arbre des ORF , seuillongueur : longueur minimale des ORF ; minGC3 et maxGC3 : limite du taux des GC3 / retour : la fonction modifie l'arbre des ORF / Expl : la fonction parcourt l'arbre des ORF et filtre les  ORF en fonction de leur long et leur taux de GC3 */

//1.3 Filtrage selon la présence de RBS
int contientRBS(char *seq) {
    // Vérifier la présence de motifs RBS
    return (strstr(seq, "AGGA") != NULL || strstr(seq, "GGAGG") != NULL || strstr(seq, "AGGAGG") != NULL);
}

void filtrerORF_RBS(tyABR *abr) {
    if (abr == NULL) return;
	
    
    // poyr filtrer l'ORF par la présence d'un RBS
    if (contientRBS(abr->pS->seq)) {
        abr->pS->hasRBS = 1;  // Marquer l'ORF comme ayant un RBS
    } else {
        abr->pS->hasRBS = 0;  // Marquer l'ORF comme sans RBS
    }
	
	
    
    // Appels récursifs pour les sous-arbres
    filtrerORF_RBS(abr->pFG);
    filtrerORF_RBS(abr->pFD);
}


