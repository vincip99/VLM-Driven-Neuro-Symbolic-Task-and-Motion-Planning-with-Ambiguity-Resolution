(define (domain manipulation)
 (:requirements :strips :typing)
 (:types robot obj location)
 (:predicates
   (holding ?ob)
   (on-table ?ob)
   (on ?ob ?pos)
 )

 (:action pick
   :parameters (?ob - obj)
   :precondition (and (not (holding ?ob)) (on-table ?ob))
   :effect (and (holding ?ob) (not (on-table ?ob)))
 )

 (:action place
   :parameters (?ob - obj ?pos - location)
   :precondition (and (holding ?ob))
   :effect (and (not (holding ?ob)) (on ?ob ?pos))
 )
)