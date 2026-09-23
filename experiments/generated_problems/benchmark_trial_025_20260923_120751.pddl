(define (problem manipulation-task)
  (:domain manipulation)
  (:objects
    yellow_can - obj
    sorting_bin - location
  )
  (:init
    (on-table yellow_can)
  )
  (:goal (and
    (on yellow_can sorting_bin)
  ))
)
