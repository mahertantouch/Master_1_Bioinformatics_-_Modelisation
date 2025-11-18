#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "LireFasta.h"       // pour declarer la fonction LireFastaSimple
#include "abr.h"             // poyr delcaration de la fonction findORF_ABR 
#include "ManipSeqSimple.h"  // declaration autres fonctions liées aux séquences et aux ORF

// autres fonctions nécessaires (de filtrage, etc)
extern void ecrire_caracteristiques(tyABR *abr, const char *nomFichier);
extern void comparerORFandCDS(tyABR *abrORF, tyABR *abrCDS);
extern void filtrerORF(tyABR *abr, int minLg, float minGC3, float maxGC3);
extern void filtrerORF_RBS(tyABR *abr);


//fction principale
int main() {
    //lecture fichier Fasta des ORF (brin direct)
    tyABR *orf = findORF_ABR(LireFastaSimple("NC_020075.fna"));  //
    if (orf == NULL) {
        printf("il y a erreur lors de la lecture du fichier NC_020075.fna.\n");
        return -1;
    }

    //fichier Fasta des CDS (généralement pour les gènes)
    tyABR *cds = LireFastaMul_ABR("NC_020075.ffn");  // Chargement des CDS
    if (cds == NULL) {
        printf("Erreur lecture des CDS dans NC_020075.ffn.\n");
        libererABR(orf);
        return -1;
    }

    //ORF sur le brin complémentaire
    tyABR *orf_complementaire = findORF_ABR(LireFastaSimple("NC_020075.fna"));
    if (orf_complementaire == NULL) {
        printf("erreur des ORFs sur le brin complzentaire.\n");
        libererABR(orf);
        libererABR(cds);
        return -1;
    }

    //cimparaison des ORF et CDS 
    comparerORFandCDS(orf, cds);  //comparaison ORF et CDS 
    comparerORFandCDS(orf_complementaire, cds);  //ciparaison ORF compl. avec CDS

    //czractéristiques des ORF et des CDS dans des fichiers
    ecrire_caracteristiques(orf, "caracteristiques_orf.txt");
    ecrire_caracteristiques(cds, "caracteristiques_cds.txt");

    //poyr filtrer des ORFs (on se base sur la taille min et la proportion minimale de GC3)
    filtrerORF(orf, 150, 0.34, 1);  // cf MesFonctions.txt
    int orfNonFiltrees = compterNoeudsNonFiltres(orf);  // Compte les ORFs non filtrées
    printf("Nombre d'ORFs non filtrées : %d\n", orfNonFiltrees);

    //in liberer la memoire des arbres
    libererABR(orf);
    libererABR(cds);
    libererABR(orf_complementaire);

    return 0;
}



