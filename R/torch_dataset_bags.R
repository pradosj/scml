


#' @title bag_sampling_dataset
#' @description Given labelled bags of instances, randomly generate new bags bag_size by random sampling random sample instances
#' @param x a generic 2D matrix object, with elements in rows and features as column
#' @param bags a list of integer vectors referencing rows of x
#' @param y a vector of target values for each bag, so the length must match nlevels(sample_ids)
#' @param nbag number of random bags to generate in the dataset
#' @param bag_size size of the bags to generate
#' @param seed set seed for random sampling
#' @param replace a logical, if TRUE the same sample might be sampled multiple time.
#' @param replace_elt a logical, if TRUE the same elements might be sampled multiple time.
#' @param post_process a function to use to post_process the batches
#' @param balancing a grouping factor for balancing
#' @import torch
#' @importFrom tibble tibble rowid_to_column
#' @importFrom dplyr mutate inner_join select slice_sample group_by ungroup slice summarize n if_else
#' @importFrom purrr map map2
#' @importFrom tidyr unnest
#' @export
#' @examples
#' bag_sampling_dataset(
#'   matrix((1:1000-1)%%100+1,100,10),
#'   split(seq(100),gl(20,5)),
#'   gl(2,10),
#'   nbag = 7L,bag_size=5L,balanced=TRUE
#' )[1:3]
bag_sampling_dataset <- torch::dataset(
  name = "bag_sampling_dataset",

  initialize = function(x,bags,y,nbag=1000L,bag_size=100L,seed=1234L,replace=TRUE,replace_elt=replace,post_process=identity,balancing=TRUE) {
    stopifnot(identical(length(y),length(bags)))
    stopifnot("some integers in bags are out of range" = max(unlist(bags))<=nrow(x))
    stopifnot("some integers in bags are out of range" = min(unlist(bags))>=1L)
    stopifnot(all(!is.na(y)))


    if (rlang::is_true(balancing)) {
      balancing <- y
    } else if (rlang::is_false(balancing)) {
      balancing <- rep_along(y,"all")
    } else {
      balancing <- as_factor(balancing)
      stopifnot(identical(length(y),length(balancing)))
    }

    self$post_process <- post_process
    self$x <- x
    self$input_bags <- tibble(
      input_bag_idx = seq_along(bags),
      y = y,
      balancing = balancing,
      elements = unname(bags)
    )

    #-#-#-#-#-#-#-#-#
    # Build bags
    #-#-#-#-#-#-#-#-#

    # First assign one sample to each bag
    set.seed(seed)
    self$bags <- self$input_bags |>
      group_by(balancing) |>
      slice_sample(n=nbag,replace=replace) %>%
      ungroup() %>%
      slice_sample(prop=1) %>%
      rowid_to_column("bag_id")

    # Then randomly select elements from selected sample
    self$bags <- self$bags %>%
      select(bag_id,elements) %>%
      unnest_longer(elements) %>%
      group_by(bag_id) %>%
      slice_sample(n=bag_size,replace=replace_elt) %>%
      summarise(elements=list(elements)) %>%
      inner_join(select(self$bags,bag_id,y,input_bag_idx),by="bag_id",relationship="one-to-one")
  },
  .length = function() {
    nrow(self$bags)
  },
  .getbatch = function(index) {
    B <- dplyr::slice(self$bags,index)
    list(
      x = local({
        x <- self$x[unlist(B$elements,use.names = FALSE),] |>
          as.matrix() |>
          torch_tensor() |>
          torch_reshape(c(nrow(B),-1,ncol(self$x)))
      }),
      y = torch_tensor(B$y)
    ) |> self$post_process()
  },
  .getitem = function(index) {
    if (is.list(index)) {index <- unlist(index)}
    self$.getbatch(index)
  }
)

