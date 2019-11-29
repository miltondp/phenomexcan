drop table if exists phenotype_info;

create table phenotype_info (
    full_code varchar(210) not null,
    short_code varchar(55) not null,
    description varchar(210) null,
    unique_description varchar(225) not null,
    type varchar(15) not null,
    n real not null,
    n_cases real null,
    n_controls real null,
    source varchar(20) not null,
    primary key(full_code)
);

-- the file here is the uncompressed version of the file generated by the jupyter
-- notebook 100_postprocessing/20_traits_and_genes_info_files.ipynb
\copy phenotype_info from /mnt/phenomexcan_base/deliverables/phenotypes_info.tsv with delimiter as E'\t' csv header;
