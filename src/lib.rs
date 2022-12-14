#![deny(missing_docs, missing_crate_level_docs)]

//! The gym-rs crate is a pure rust implementation of OpenAI's Gym

extern crate log;

mod action_type;
mod cart_pole;
mod gif_render;
mod gym_env;
mod utils;

pub use action_type::ActionType;
pub use cart_pole::CartPoleEnv;
pub use gif_render::GifRender;
pub use gym_env::GymEnv;
pub use utils::scale;
