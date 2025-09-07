use crate::{Generate, Query, Repeat};

#[derive(Default)]
pub struct PlanOptions {
    #[cfg(feature = "rag")]
    pub aug: crate::augment::AugmentOptions,
}

async fn plan_vec_iter(v: &[Query], po: &PlanOptions) -> anyhow::Result<Vec<Query>> {
    // TODO: this can't be the most efficient way to do this
    let results = futures::future::try_join_all(v.iter().map(|u| plan_iter(u, po))).await?;
    let iter = results.into_iter().flatten();
    Ok(iter.collect())
}

fn cross_if_needed(v: Vec<Query>) -> Query {
    match &v[..] {
        [query] => query.clone(),
        _ => Query::Cross(v),
    }
}

#[async_recursion::async_recursion]
async fn plan_iter(query: &Query, po: &PlanOptions) -> anyhow::Result<Vec<Query>> {
    // this is probably the wrong place for this, but here we expand any Repeats under Plus or Cross
    match query {
        Query::Plus(v) => Ok(vec![Query::Plus(plan_vec_iter(v, po).await?)]),
        Query::Cross(v) => Ok(vec![Query::Cross(plan_vec_iter(v, po).await?)]),

        #[cfg(feature = "rag")]
        Query::Augment(a) => {
            let retrieved = crate::augment::retrieve(&a.embedding_model, &a.body, &a.doc, &po.aug)
                .await?;
            let iter = retrieved.map(|s| Query::Message(crate::Message::User(s)));
            Ok(vec![Query::Plus(iter.collect())])
        }

        Query::Repeat(Repeat { n, query }) => {
            let q = plan_iter(query, po).await?;
            let iter = ::std::iter::repeat_n(q, *n).flatten();
            Ok(iter.collect())
        }

        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
        }) => Ok(vec![Query::Generate(Generate {
            model: model.clone(),
            input: Box::new(cross_if_needed(plan_iter(input, po).await?)),
            max_tokens: *max_tokens,
            temperature: *temperature,
        })]),

        otherwise => Ok(vec![otherwise.clone()]),
    }
}

/// This tries to remove some unnecessary syntactic complexities,
/// e.g. Plus-of-Plus or Cross with a tail Cross.
fn simplify(query: &Query) -> Query {
    match query {
        Query::Plus(v) => Query::Plus({
            let iter: Box<dyn Iterator<Item = Query>> = match &v[..] {
                // Plus of Plus
                [Query::Plus(v2)] => Box::new(v2.iter().map(simplify)),
                otherwise => Box::new(otherwise.iter().map(simplify)),
            };
            iter.collect()
        }),
        Query::Cross(v) => Query::Cross({
            let iter: Box<dyn Iterator<Item = Query>> = match &v[..] {
                // Cross of tail Cross
                [.., Query::Cross(v2)] => Box::new(v
                    .iter()
                    .take(v.len() - 1)
                    .chain(v2.iter())
                    .map(simplify)),
                otherwise => Box::new(otherwise.iter().map(simplify)),
            };
            iter.collect()
        }),
        Query::Generate(Generate {
            model,
            input,
            max_tokens,
            temperature,
        }) => Query::Generate(Generate {
            model: model.clone(),
            input: Box::new(simplify(input)),
            max_tokens: *max_tokens,
            temperature: *temperature,
        }),
        otherwise => otherwise.clone(),
    }
}

pub async fn plan(query: &Query, po: &PlanOptions) -> anyhow::Result<Query> {
    #[cfg(feature = "rag")]
    crate::augment::index(query, &po.aug).await?;

    Ok(simplify(&cross_if_needed(plan_iter(query, po).await?)))
}
