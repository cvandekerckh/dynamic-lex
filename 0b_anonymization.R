library(tidyverse)

# Fun
anon <- function (x) {
  tmp_dt <- unlist(strsplit(x, split = ","))
  tmp_dt[1] <- ids[tmp_dt[1]]
  paste(tmp_dt, collapse = ",")
}


# VPL --------------------------------------------------------------------------
# garder les colonnes du codebook + Twitter_ID (mai avec la valeur de VPL_ID)
dir.create("../data/ids_match")

# Matching files to process
files <- sort(dir("../data/raw/vpl", pattern = "ended.csv", full.names = TRUE))

files_usrs <- files[!grepl("politicians", files)]
files_pols <- files[grepl("politicians", files)]
codebooks  <- dir("../data/raw/vpl", pattern = "codebook", full.names = TRUE)

for (item in seq(files_usrs)) {

message("Starting: ", files[item])
  data_ext_usrs <- read_csv(files_usrs[item])
  data_ext_pols <- read_csv(files_pols[item])
  cb_ext        <- read_csv(codebooks[item])

# Output filename
  filename <- gsub("..*/", "", files_usrs[item]) %>%
    sub(".csv", "_match.csv", .)

# Saving match file
  file_match <- file.path("../data/ids_match", filename)
  write_csv(data_ext_usrs[c('Twitter_ID', 'VPL_ID')], file_match)

# Filtering and replacing values
  data_ext_usrs <- data_ext_usrs[c("Twitter_ID", "VPL_ID", cb_ext[['field_local']])]
  data_ext_usrs['Twitter_ID'] <- data_ext_usrs['VPL_ID']
  data_ext_pols <- data_ext_pols[c("VPL_ID", "Twitter_ID", "Twitter_account", "Party")]

  write_csv(data_ext_usrs, files_usrs[item])
  write_csv(data_ext_pols, files_pols[item])


# Follow Network ---------------------------------------------------------------
  files <- sort(dir("../data/raw/follow/", recursive = TRUE,
               pattern = "network", full.names = TRUE))


  original_ids <- read_csv(file_match) %>% na.omit
  ids <- setNames(original_ids$VPL_ID,
                  as.character(original_ids$Twitter_ID))

  tmp_file <- files[item]
  data_ntw <- readLines(tmp_file)

  data_ntw[2:length(data_ntw)] <- sapply(data_ntw[2:length(data_ntw)], anon)
  con <- file(tmp_file)
  writeLines(data_ntw, con)
  close(con)


# Text Link --------------------------------------------------------------------
  files_link <- sort(dir("../data/raw/text/", recursive = TRUE,
                    pattern = "link", full.names = TRUE))
  files_link <- sort(files_link[!grepl("politicians", files_link)])

  tmp_file <- files_link[item]
  data_txt <- read_csv(tmp_file)
  data_txt[c('Twitter_ID', 'Twitter_account')] <- data_txt['VPL_ID']

  write_csv(data_txt, tmp_file)

}
