/// Defines the type of action to take in the environment
#[derive(Copy, Clone, Debug)]
pub enum ActionType {
    /// A discrete action
    Discrete(u32),
    // /// A continuous action
    // Continuous(Vec<f64>),
}
